from __future__ import annotations

import numpy as np


def cwt_scalogram(
    t: np.ndarray,
    x: np.ndarray,
    *,
    fmin: float = 1.0,
    fmax: float = 60.0,
    n_freqs: int = 180,
    wavelet: str = "morl",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a CWT scalogram on a linear frequency grid."""

    import pywt

    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    dt = float(np.median(np.diff(t)))
    fs = 1.0 / dt
    freqs = np.linspace(fmin, fmax, n_freqs)
    scales = pywt.central_frequency(wavelet) * fs / freqs
    coeffs, _ = pywt.cwt(x, scales, wavelet, sampling_period=dt)
    return freqs, scales, np.abs(coeffs)


def ridge_frequency(freqs: np.ndarray, power: np.ndarray) -> np.ndarray:
    """Return the max-power frequency for each time sample."""

    idx = np.argmax(power, axis=0)
    return np.asarray(freqs)[idx]


def dwt_energy(x: np.ndarray, *, wavelet: str = "db4", level: int | None = None):
    """Return DWT coefficients, labels, and energy per scale."""

    import pywt

    x = np.asarray(x, dtype=float)
    if level is None:
        level = pywt.dwt_max_level(len(x), pywt.Wavelet(wavelet).dec_len)
    coeffs = pywt.wavedec(x, wavelet, level=level)
    labels = [f"A{level}"] + [f"D{level - i + 1}" for i in range(1, level + 1)]
    energy = np.array([np.sum(c**2) for c in coeffs], dtype=float)
    return coeffs, labels, energy


def reconstruct_dominant_mode(
    x: np.ndarray,
    *,
    wavelet: str = "db4",
    level: int | None = None,
) -> tuple[np.ndarray, str, np.ndarray]:
    """Reconstruct the signal using only the highest-energy DWT scale."""

    import pywt

    coeffs, labels, energy = dwt_energy(x, wavelet=wavelet, level=level)
    idx = int(np.argmax(energy))
    selected = [np.zeros_like(c) for c in coeffs]
    selected[idx] = coeffs[idx]
    rec = pywt.waverec(selected, wavelet)[: len(x)]
    return rec, labels[idx], energy

