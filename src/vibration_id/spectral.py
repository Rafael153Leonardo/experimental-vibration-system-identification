from __future__ import annotations

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import get_window, welch


def sampling_rate(t: np.ndarray) -> float:
    """Estimate sampling rate from a time vector."""

    t = np.asarray(t, dtype=float)
    dt = float(np.median(np.diff(t)))
    return 1.0 / dt


def compute_fft(
    t: np.ndarray,
    x: np.ndarray,
    *,
    window: str | None = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    """Return positive FFT frequencies and amplitude-calibrated magnitudes.

    A window (Hann by default) is applied to reduce spectral leakage. Amplitudes
    are normalized by the window's coherent gain ``sum(w)`` so that a pure tone
    keeps its true amplitude; pass ``window=None`` for the legacy rectangular
    behavior.
    """

    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(x)
    x = x - np.mean(x)
    if window is None:
        w = np.ones(n)
    else:
        w = get_window(window, n)
    coherent_gain = float(np.sum(w))
    y = rfft(x * w)
    freqs = rfftfreq(n, float(np.median(np.diff(t))))
    amp = 2.0 * np.abs(y) / coherent_gain
    if len(amp):
        amp[0] *= 0.5
    return freqs, amp


def _parabolic_peak(freqs: np.ndarray, amp: np.ndarray, idx: int) -> float:
    """Sub-bin peak frequency via parabolic interpolation around bin ``idx``."""

    if idx <= 0 or idx >= len(amp) - 1:
        return float(freqs[idx])
    a, b, c = amp[idx - 1], amp[idx], amp[idx + 1]
    denom = a - 2.0 * b + c
    if denom == 0:
        return float(freqs[idx])
    delta = 0.5 * (a - c) / denom
    df = float(freqs[1] - freqs[0])
    return float(freqs[idx] + delta * df)


def dominant_frequency(
    t: np.ndarray,
    x: np.ndarray,
    *,
    fmin: float = 0.1,
    fmax: float | None = None,
    window: str | None = "hann",
    interpolate: bool = True,
) -> float:
    """Estimate the dominant frequency from the FFT peak.

    With ``interpolate=True`` the peak is refined to sub-bin resolution by
    parabolic interpolation, which matters for short crops where the bin spacing
    is coarse.
    """

    freqs, amp = compute_fft(t, x, window=window)
    mask = freqs >= fmin
    if fmax is not None:
        mask &= freqs <= fmax
    if not np.any(mask):
        raise ValueError("No frequency bin inside the requested range.")
    masked_idx = np.argmax(amp[mask])
    global_idx = int(np.flatnonzero(mask)[masked_idx])
    if interpolate:
        return _parabolic_peak(freqs, amp, global_idx)
    return float(freqs[global_idx])


def compute_psd(
    t: np.ndarray,
    x: np.ndarray,
    *,
    nperseg: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Welch power spectral density."""

    fs = sampling_rate(t)
    freqs, psd = welch(
        np.asarray(x, dtype=float),
        fs=fs,
        window="hann",
        nperseg=nperseg,
        scaling="density",
        detrend="constant",
    )
    return freqs, psd
