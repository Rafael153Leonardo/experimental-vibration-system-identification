from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import hilbert
from scipy.stats import linregress


@dataclass(frozen=True)
class DampingFit:
    amplitude0: float
    gamma: float
    tau: float
    half_life: float
    r_squared: float
    envelope: np.ndarray
    model: np.ndarray


def hilbert_envelope(x: np.ndarray) -> np.ndarray:
    """Return the analytic-signal amplitude envelope."""

    return np.abs(hilbert(np.asarray(x, dtype=float)))


def fit_exponential_envelope(
    t: np.ndarray,
    x: np.ndarray,
    *,
    tmin: float = 0.0,
    tmax: float | None = None,
    min_envelope: float = 1e-12,
) -> DampingFit:
    """Fit A(t) = A0 exp(-gamma t / 2) to the Hilbert envelope."""

    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    env = hilbert_envelope(x)

    if tmax is None:
        tmax = float(t[0] + 0.3 * (t[-1] - t[0]))

    mask = (t >= tmin) & (t <= tmax) & (env > min_envelope)
    if np.count_nonzero(mask) < 3:
        raise ValueError("Not enough samples for envelope fit.")

    slope, intercept, r_value, _, _ = linregress(t[mask], np.log(env[mask]))
    gamma = -2.0 * float(slope)
    amplitude0 = float(np.exp(intercept))
    model = amplitude0 * np.exp(-gamma * t / 2.0)

    return DampingFit(
        amplitude0=amplitude0,
        gamma=gamma,
        tau=2.0 / gamma if gamma != 0 else np.inf,
        half_life=2.0 * np.log(2.0) / gamma if gamma != 0 else np.inf,
        r_squared=float(r_value**2),
        envelope=env,
        model=model,
    )


def quality_factor(frequency_hz: float, gamma: float) -> float:
    """Physical quality factor of a damped oscillator.

    For an envelope ``A0 exp(-gamma t / 2)`` the standard quality factor is
    ``Q = omega_n / gamma = 2*pi*f0 / gamma``. An earlier version of this
    function returned ``f0 / gamma``, which is off by a factor of ``2*pi`` and is
    *not* the physical Q; see :func:`frequency_to_decay_ratio` for that quantity.
    """

    if gamma == 0:
        return float("inf")
    return float(2.0 * np.pi * frequency_hz / gamma)


def frequency_to_decay_ratio(frequency_hz: float, gamma: float) -> float:
    """Dimensionless ratio ``f0 / gamma`` (the old, non-standard convention)."""

    if gamma == 0:
        return float("inf")
    return float(frequency_hz / gamma)


def quality_factor_half_power(
    freqs: np.ndarray,
    amplitudes: np.ndarray,
    *,
    peak_index: int | None = None,
) -> tuple[float, float, float]:
    """Estimate Q from the -3 dB (half-power) bandwidth around a spectral peak.

    Ported from the original ``fatorQ.py``. Returns ``(q, peak_frequency_hz,
    bandwidth_hz)`` where ``Q = f_r / delta_f`` and ``delta_f`` is the width of
    the band where the amplitude first drops below ``peak / sqrt(2)`` on each
    side of the peak. This is the physically standard, spectrum-based estimator
    and does not depend on a separate damping fit.
    """

    freqs = np.asarray(freqs, dtype=float)
    amplitudes = np.asarray(amplitudes, dtype=float)
    if len(freqs) != len(amplitudes) or len(freqs) < 3:
        raise ValueError("freqs and amplitudes must be equal-length arrays with >= 3 bins.")

    idx = int(np.argmax(amplitudes)) if peak_index is None else int(peak_index)
    peak_amp = amplitudes[idx]
    if peak_amp <= 0:
        raise ValueError("Peak amplitude must be positive.")

    half_power = peak_amp / np.sqrt(2.0)
    left = np.where(amplitudes[:idx] <= half_power)[0]
    right = np.where(amplitudes[idx:] <= half_power)[0]
    if left.size == 0 or right.size == 0:
        raise ValueError("Half-power points not found inside the provided spectrum.")

    f1 = freqs[left[-1]]
    f2 = freqs[right[0] + idx]
    bandwidth = float(f2 - f1)
    f_r = float(freqs[idx])
    if bandwidth <= 0:
        raise ValueError("Non-positive half-power bandwidth.")
    return float(f_r / bandwidth), f_r, bandwidth
