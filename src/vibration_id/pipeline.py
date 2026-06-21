"""Shared signal-loading pipeline used by the command-line scripts.

Every analysis script repeated the same boilerplate: load a CSV, remove the DC
offset from the tail, crop to the vibration onset and wavelet-denoise. This
module centralizes that so the scripts stay thin and the preprocessing is
identical everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vibration_id.io import load_signal_csv
from vibration_id.preprocessing import crop_from_onset, remove_dc_offset, wavelet_denoise


@dataclass(frozen=True)
class CleanedSignal:
    """Result of the standard preprocessing pipeline."""

    t: np.ndarray
    raw: np.ndarray  # DC-removed and onset-cropped, but not denoised
    clean: np.ndarray  # additionally wavelet-denoised
    onset: int


def load_clean_signal(
    path: str | Path,
    *,
    time_col: str | None = None,
    signal_col: str | None = None,
    dc_mode: str = "tail",
    crop: bool = True,
    safety_samples: int = 5,
    denoise: bool = True,
    wavelet: str = "db8",
    level: int = 2,
) -> CleanedSignal:
    """Load a CSV and apply the standard preprocessing pipeline.

    Returns both the ``raw`` (DC-removed, cropped) and ``clean`` (denoised)
    signals so callers can use whichever they need without re-running the steps.
    """

    data = load_signal_csv(path, center_signal=False, time_col=time_col, signal_col=signal_col)
    t = data["time_s"].to_numpy()
    x = remove_dc_offset(data["signal"].to_numpy(), mode=dc_mode)

    onset = 0
    if crop:
        t, x, onset = crop_from_onset(t, x, safety_samples=safety_samples)

    clean = wavelet_denoise(x, wavelet=wavelet, level=level) if denoise else x
    return CleanedSignal(t=t, raw=x, clean=clean, onset=onset)


def decimate(t: np.ndarray, x: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Evenly subsample ``(t, x)`` to at most ``max_points`` samples."""

    if max_points <= 0 or len(t) <= max_points:
        return t, x
    idx = np.linspace(0, len(t) - 1, max_points).astype(int)
    return t[idx], x[idx]
