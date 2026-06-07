from __future__ import annotations

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import welch


def sampling_rate(t: np.ndarray) -> float:
    """Estimate sampling rate from a time vector."""

    t = np.asarray(t, dtype=float)
    dt = float(np.median(np.diff(t)))
    return 1.0 / dt


def compute_fft(t: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return positive FFT frequencies and normalized amplitudes."""

    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    dt = float(np.median(np.diff(t)))
    y = rfft(x - np.mean(x))
    freqs = rfftfreq(len(x), dt)
    amp = 2.0 * np.abs(y) / len(x)
    if len(amp):
        amp[0] *= 0.5
    return freqs, amp


def dominant_frequency(
    t: np.ndarray,
    x: np.ndarray,
    *,
    fmin: float = 0.1,
    fmax: float | None = None,
) -> float:
    """Estimate the dominant frequency from the FFT peak."""

    freqs, amp = compute_fft(t, x)
    mask = freqs >= fmin
    if fmax is not None:
        mask &= freqs <= fmax
    if not np.any(mask):
        raise ValueError("No frequency bin inside the requested range.")
    idx = np.argmax(amp[mask])
    return float(freqs[mask][idx])


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

