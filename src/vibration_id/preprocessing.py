from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def remove_dc_offset(x: np.ndarray, *, mode: str = "mean", tail_fraction: float = 0.2) -> np.ndarray:
    """Remove signal offset using either the full mean or a tail mean."""

    x = np.asarray(x, dtype=float)
    if mode == "mean":
        offset = np.mean(x)
    elif mode == "tail":
        n_tail = max(1, int(len(x) * tail_fraction))
        offset = np.mean(x[-n_tail:])
    else:
        raise ValueError("mode must be 'mean' or 'tail'")
    return x - offset


def detect_onset_by_gradient(x: np.ndarray, *, safety_samples: int = 5) -> int:
    """Find the approximate vibration start by the largest gradient jump."""

    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        return 0
    onset = int(np.argmax(np.abs(np.gradient(x))))
    return min(len(x) - 1, onset + max(0, safety_samples))


def crop_from_onset(
    t: np.ndarray,
    x: np.ndarray,
    *,
    safety_samples: int = 5,
    reset_time: bool = True,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Crop the initial dead time before the vibration event."""

    idx = detect_onset_by_gradient(x, safety_samples=safety_samples)
    t_crop = np.asarray(t, dtype=float)[idx:]
    x_crop = np.asarray(x, dtype=float)[idx:]
    if reset_time and len(t_crop):
        t_crop = t_crop - t_crop[0]
    return t_crop, x_crop, idx


def wavelet_denoise(
    x: np.ndarray,
    *,
    wavelet: str = "db8",
    level: int = 2,
    mode: str = "soft",
) -> np.ndarray:
    """Denoise a signal with universal-threshold wavelet shrinkage."""

    import pywt

    x = np.asarray(x, dtype=float)
    coeffs = pywt.wavedec(x, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2.0 * np.log(len(x)))
    filtered = [coeffs[0]] + [pywt.threshold(c, threshold, mode=mode) for c in coeffs[1:]]
    return pywt.waverec(filtered, wavelet)[: len(x)]


def velocity_savgol(
    t: np.ndarray,
    x: np.ndarray,
    *,
    window_length: int = 81,
    polyorder: int = 3,
) -> np.ndarray:
    """Estimate velocity with a Savitzky-Golay derivative."""

    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    dt = float(np.median(np.diff(t)))
    if window_length % 2 == 0:
        window_length += 1
    window_length = min(window_length, len(x) - (1 - len(x) % 2))
    if window_length <= polyorder:
        return np.gradient(x, dt)
    return savgol_filter(x, window_length, polyorder, deriv=1, delta=dt)

