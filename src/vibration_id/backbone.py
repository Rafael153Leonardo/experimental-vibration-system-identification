"""Backbone-curve and damping-law diagnostics for a free ring-down.

The free-decay records answer two questions the headline modulus analysis never
asks, and that settle where the system's nonlinearity actually lives:

1. **Is the stiffness nonlinear?**  A Duffing stiffness makes the oscillation
   frequency depend on amplitude -- the *backbone curve*.  To first order
   ``f(A) = f0 (1 + kappa * A**2)``.  Reading the instantaneous frequency and
   amplitude off the analytic (Hilbert) signal maps ``f`` against ``A``
   directly.  A **flat** backbone means the stiffness is linear, and therefore
   that any cubic stiffness term recovered by SINDy or the global Duffing fit is
   an artifact of the sensor output map, not beam physics.

2. **Is the damping linear?**  A single viscous term gives an exponential
   envelope ``A(t) = r0 * exp(-gamma * t / 2)``.  Amplitude-dependent (drag-like)
   damping adds an ``eta * A**2`` term, whose closed-form envelope is
   :func:`vibration_id.global_fit.nonlinear_envelope`.  Fitting both on the same
   envelope and comparing the coefficient of determination shows which law the
   data prefer.

Everything here runs on numpy/scipy alone and reuses the envelope law already
defined in :mod:`vibration_id.global_fit`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks, hilbert, savgol_filter

from vibration_id.global_fit import fit_nonlinear_envelope, nonlinear_envelope


@dataclass(frozen=True)
class BackboneFit:
    """Amplitude dependence of the instantaneous frequency.

    ``f(A) = f0_hz + slope_hz_per_amp2 * A**2`` fitted over the ring-down.
    ``fractional_pull`` is the relative frequency change across the observed
    amplitude span; ``pearson_r`` is the correlation of the instantaneous
    frequency with ``A**2`` (near zero => no stiffness nonlinearity).
    """

    f0_hz: float
    slope_hz_per_amp2: float
    kappa_per_amp2: float
    fractional_pull: float
    pearson_r: float
    amplitude_lo: float
    amplitude_hi: float


@dataclass(frozen=True)
class DampingComparison:
    """Linear-viscous vs amplitude-dependent damping, fitted to one envelope."""

    r0: float
    gamma_linear: float
    q_linear: float
    r2_linear: float
    gamma_nonlinear: float
    eta_nonlinear: float
    r2_nonlinear: float

    @property
    def prefers_nonlinear(self) -> bool:
        """True when the amplitude-dependent law explains more of the envelope."""

        return self.r2_nonlinear > self.r2_linear


@dataclass(frozen=True)
class RingdownAnalysis:
    """Bundle of both diagnostics plus the arrays needed to plot them."""

    backbone: BackboneFit
    damping: DampingComparison
    t_window: np.ndarray
    envelope: np.ndarray
    linear_model: np.ndarray
    nonlinear_model: np.ndarray
    amplitude: np.ndarray
    inst_frequency: np.ndarray
    amplitude_bins: np.ndarray
    frequency_bins: np.ndarray
    frequency_bin_err: np.ndarray


def _odd_window(fs: float, frequency_hz: float, *, periods: float, floor: int) -> int:
    """Odd Savitzky-Golay window spanning ``periods`` cycles (at least ``floor``)."""

    win = int(fs * periods / max(frequency_hz, 1e-9))
    win = max(win, floor)
    return win | 1


def _smooth(y: np.ndarray, fs: float, frequency_hz: float, *, periods: float) -> np.ndarray:
    """Savitzky-Golay smoothing over ``periods`` oscillation cycles.

    The analytic amplitude and instantaneous frequency of a decaying tone wobble
    at the carrier when the signal carries harmonics (here, the sensor's cubic
    output map); averaging over a cycle recovers the meaningful slow envelope.
    """

    y = np.asarray(y, dtype=float)
    win = _odd_window(fs, frequency_hz, periods=periods, floor=51)
    win = min(win, (len(y) - 1) | 1)
    if win <= 2:
        return y
    return savgol_filter(y, win, 2)


def _peak_envelope(t: np.ndarray, x: np.ndarray, *, frequency_hz: float) -> np.ndarray:
    """Symmetric ring-down envelope from interpolated cycle peaks.

    Cleaner than the analytic amplitude ``|hilbert(x)|`` when the record carries
    harmonics or a slow asymmetry (the sensor's cubic map imprints both), which
    make ``|hilbert(x)|`` wobble at the carrier. Interpolating the upper (positive
    peaks) and lower (negative peaks) envelopes separately and averaging them also
    removes the per-half-cycle alternation that asymmetry produces.
    """

    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    fs = 1.0 / float(np.median(np.diff(t)))
    distance = max(int(0.8 * fs / max(frequency_hz, 1e-9)), 1)  # ~1 period between same-sign peaks
    pos, _ = find_peaks(x, distance=distance)
    neg, _ = find_peaks(-x, distance=distance)
    if len(pos) < 3 or len(neg) < 3:
        return np.abs(hilbert(x))
    upper = np.interp(t, t[pos], x[pos])
    lower = np.interp(t, t[neg], -x[neg])
    return 0.5 * (upper + lower)


def _bin_backbone(
    amplitude: np.ndarray, inst_freq: np.ndarray, *, n_bins: int = 16
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin instantaneous frequency by amplitude; return per-bin (A, mean f, std f)."""

    amplitude = np.asarray(amplitude, dtype=float)
    inst_freq = np.asarray(inst_freq, dtype=float)
    edges = np.linspace(amplitude.min(), amplitude.max(), n_bins + 1)
    idx = np.clip(np.digitize(amplitude, edges) - 1, 0, n_bins - 1)
    a_mid, f_mean, f_err = [], [], []
    for b in range(n_bins):
        sel = idx == b
        if np.count_nonzero(sel) >= 5:
            a_mid.append(0.5 * (edges[b] + edges[b + 1]))
            f_mean.append(float(np.mean(inst_freq[sel])))
            f_err.append(float(np.std(inst_freq[sel])))
    return np.asarray(a_mid), np.asarray(f_mean), np.asarray(f_err)


def instantaneous_amplitude_frequency(
    t: np.ndarray,
    x: np.ndarray,
    *,
    frequency_hz: float,
    smooth_periods: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the analytic-signal amplitude and instantaneous frequency (Hz).

    The instantaneous frequency is the Savitzky-Golay derivative of the unwrapped
    analytic phase, smoothed over ``smooth_periods`` oscillation cycles to reject
    the per-sample phase jitter of a noisy record.
    """

    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    fs = 1.0 / float(np.median(np.diff(t)))
    z = hilbert(x)
    amplitude = np.abs(z)
    phase = np.unwrap(np.angle(z))
    win = _odd_window(fs, frequency_hz, periods=smooth_periods, floor=51)
    if win >= len(x):
        win = (len(x) - 1) | 1
    if win <= 2:
        inst_freq = np.gradient(phase, 1.0 / fs) / (2.0 * np.pi)
    else:
        inst_freq = savgol_filter(phase, win, 2, deriv=1, delta=1.0 / fs) / (2.0 * np.pi)
    return amplitude, inst_freq


def _ringdown_window(
    t: np.ndarray,
    amplitude: np.ndarray,
    *,
    fs: float,
    frequency_hz: float,
    start_seconds: float,
    floor_fraction: float,
) -> slice:
    """Slice covering the clean decay: after the onset transient, above the floor."""

    win = _odd_window(fs, frequency_hz, periods=1.0, floor=101)
    smooth = savgol_filter(amplitude, min(win, (len(amplitude) - 1) | 1), 2)
    peak = int(np.argmax(smooth))
    a_peak = smooth[peak]
    below = np.where(smooth[peak:] < floor_fraction * a_peak)[0]
    end = peak + int(below[0]) if below.size else len(smooth)
    start = peak + int(start_seconds * fs)
    if end - start < int(2 * fs):  # fall back to the whole post-peak decay
        start = peak
    return slice(start, end)


def fit_backbone(
    t: np.ndarray,
    x: np.ndarray,
    *,
    frequency_hz: float,
    min_amplitude_fraction: float = 0.12,
    start_seconds: float = 0.3,
    floor_fraction: float = 0.08,
) -> tuple[BackboneFit, np.ndarray, np.ndarray]:
    """Fit ``f(A) = f0 + slope * A**2`` over the ring-down.

    Returns the fit together with the amplitude and instantaneous-frequency
    scatter that produced it (for plotting).
    """

    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    fs = 1.0 / float(np.median(np.diff(t)))
    _, inst_freq = instantaneous_amplitude_frequency(t, x, frequency_hz=frequency_hz)
    # Peak-interpolated envelope for amplitude; average out the per-cycle wobble
    # the harmonic content imprints on the analytic phase for the frequency.
    amplitude = _peak_envelope(t, x, frequency_hz=frequency_hz)
    inst_freq = _smooth(inst_freq, fs, frequency_hz, periods=2.0)
    window = _ringdown_window(
        t,
        amplitude,
        fs=fs,
        frequency_hz=frequency_hz,
        start_seconds=start_seconds,
        floor_fraction=floor_fraction,
    )
    a = amplitude[window]
    f = inst_freq[window]
    keep = a > min_amplitude_fraction * np.max(amplitude)
    a, f = a[keep], f[keep]
    if len(a) < 3:
        raise ValueError("Not enough ring-down samples for a backbone fit.")

    slope, intercept = np.polyfit(a**2, f, 1)
    a_lo, a_hi = float(np.percentile(a, 5)), float(np.percentile(a, 95))
    f_lo = intercept + slope * a_lo**2
    f_hi = intercept + slope * a_hi**2
    pearson_r = float(np.corrcoef(a**2, f)[0, 1]) if np.std(a) > 0 else 0.0

    fit = BackboneFit(
        f0_hz=float(intercept),
        slope_hz_per_amp2=float(slope),
        kappa_per_amp2=float(slope / intercept) if intercept else float("nan"),
        fractional_pull=float((f_hi - f_lo) / intercept) if intercept else float("nan"),
        pearson_r=pearson_r,
        amplitude_lo=a_lo,
        amplitude_hi=a_hi,
    )
    return fit, a, f


def _linear_envelope(t: np.ndarray, r0: float, gamma: float) -> np.ndarray:
    return r0 * np.exp(-gamma * np.asarray(t, dtype=float) / 2.0)


def _r_squared(y: np.ndarray, model: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    ss_res = float(np.sum((y - model) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def compare_damping_models(
    t: np.ndarray,
    x: np.ndarray,
    *,
    frequency_hz: float,
    start_seconds: float = 0.3,
    floor_fraction: float = 0.08,
) -> tuple[DampingComparison, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit a linear-viscous and an amplitude-dependent envelope to the ring-down.

    Both are least-squares fits on the linear-amplitude scale, so their
    coefficients of determination compare fairly. Returns the comparison plus
    ``(t_window, envelope, linear_model, nonlinear_model)`` for plotting.
    """

    from scipy.optimize import curve_fit

    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    fs = 1.0 / float(np.median(np.diff(t)))
    amplitude = _peak_envelope(t, x, frequency_hz=frequency_hz)
    window = _ringdown_window(
        t,
        amplitude,
        fs=fs,
        frequency_hz=frequency_hz,
        start_seconds=start_seconds,
        floor_fraction=floor_fraction,
    )
    tw = t[window] - t[window][0]
    env = amplitude[window]

    # Linear viscous: A(t) = r0 exp(-gamma t / 2)
    r0_guess = float(env[0])
    (r0_lin, gamma_lin), _ = curve_fit(
        _linear_envelope, tw, env, p0=[r0_guess, 0.2], bounds=([1e-9, 0.0], [np.inf, np.inf]), maxfev=20000
    )
    linear_model = _linear_envelope(tw, r0_lin, gamma_lin)

    # Amplitude-dependent: reuse the closed-form nonlinear envelope.
    nl = fit_nonlinear_envelope(tw, env, smooth_window=0)
    nonlinear_model = nonlinear_envelope(tw, nl.r0, nl.gamma, nl.eta)

    comparison = DampingComparison(
        r0=float(r0_lin),
        gamma_linear=float(gamma_lin),
        q_linear=float(2.0 * np.pi * frequency_hz / gamma_lin) if gamma_lin > 0 else float("inf"),
        r2_linear=_r_squared(env, linear_model),
        gamma_nonlinear=float(nl.gamma),
        eta_nonlinear=float(nl.eta),
        r2_nonlinear=_r_squared(env, nonlinear_model),
    )
    return comparison, tw, env, linear_model, nonlinear_model


def analyze_ringdown(
    t: np.ndarray,
    x: np.ndarray,
    *,
    frequency_hz: float,
    start_seconds: float = 0.3,
    floor_fraction: float = 0.08,
) -> RingdownAnalysis:
    """Run both diagnostics and bundle the results with the plotting arrays."""

    backbone, amp, inst_freq = fit_backbone(
        t, x, frequency_hz=frequency_hz, start_seconds=start_seconds, floor_fraction=floor_fraction
    )
    damping, tw, env, lin, nl = compare_damping_models(
        t, x, frequency_hz=frequency_hz, start_seconds=start_seconds, floor_fraction=floor_fraction
    )
    a_bins, f_bins, f_err = _bin_backbone(amp, inst_freq)
    return RingdownAnalysis(
        backbone=backbone,
        damping=damping,
        t_window=tw,
        envelope=env,
        linear_model=lin,
        nonlinear_model=nl,
        amplitude=amp,
        inst_frequency=inst_freq,
        amplitude_bins=a_bins,
        frequency_bins=f_bins,
        frequency_bin_err=f_err,
    )
