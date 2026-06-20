"""Cantilever modal analysis: separate boundary condition from material.

A clamped-free Euler-Bernoulli beam has natural frequencies at fixed ratios set
by the eigenvalues ``beta_n * L`` (1.875, 4.694, 7.855, ...), so the higher
modes are NOT integer harmonics of the fundamental -- they follow ``beta_n^2``.

That ladder is a fingerprint of the boundary condition, independent of the
Young modulus and the overall scale. Driving the same beam at several modes
therefore lets you check whether the support is an ideal clamp (measured ratios
match the ladder), and only then trust the modulus inferred from the absolute
frequencies. This is exactly how the inox ruler was diagnosed: the forced modes
(~5.0 / 31.2 / 86.8 / 173.5 Hz) match the ideal ladder within ~1%, proving an
ideal clamp and pinning the earlier discrepancy on the thickness, not the
boundary.
"""

from __future__ import annotations

import numpy as np

from vibration_id.materials import BeamGeometry, young_modulus_from_frequency

# Eigenvalues beta_n * L for a clamped-free (cantilever) Euler-Bernoulli beam.
CANTILEVER_BETA_L = (1.875104, 4.694091, 7.854757, 10.995541, 14.137168)


def cantilever_mode_ratios(n_modes: int) -> np.ndarray:
    """Ideal ``f_n / f_1`` ratios for the first ``n_modes`` (``(beta_n/beta_1)^2``)."""

    if not 1 <= n_modes <= len(CANTILEVER_BETA_L):
        raise ValueError(f"n_modes must be in 1..{len(CANTILEVER_BETA_L)}")
    b = np.array(CANTILEVER_BETA_L[:n_modes])
    return (b / b[0]) ** 2


def cantilever_mode_frequencies(fundamental_hz: float, n_modes: int) -> np.ndarray:
    """Predict the first ``n_modes`` cantilever frequencies from the fundamental."""

    return float(fundamental_hz) * cantilever_mode_ratios(n_modes)


def fundamental_from_modes(frequencies: np.ndarray, mode_numbers: list[int]) -> float:
    """Least-squares fundamental implied by measured modes ``f_n = f1 * r_n``.

    ``mode_numbers`` are the 1-based mode indices of ``frequencies`` (e.g.
    ``[2, 3, 4]`` for the higher modes only). Robust because every mode votes on
    the same ``f1``.
    """

    frequencies = np.asarray(frequencies, dtype=float)
    if len(frequencies) != len(mode_numbers):
        raise ValueError("frequencies and mode_numbers must have equal length.")
    ratios = cantilever_mode_ratios(max(mode_numbers))
    r = np.array([ratios[m - 1] for m in mode_numbers])
    return float(np.sum(r * frequencies) / np.sum(r * r))


def modal_ratio_deviation(frequencies: np.ndarray, mode_numbers: list[int]) -> float:
    """RMS relative deviation of measured ratios from the ideal clamp ladder.

    Near 0 means an ideal clamp with no significant tip mass. A soft rotational
    clamp pushes higher-mode ratios *up*; a tip mass pushes them *up* too, so a
    clean match to the ladder rules both out.
    """

    frequencies = np.asarray(frequencies, dtype=float)
    f1 = fundamental_from_modes(frequencies, mode_numbers)
    ratios = cantilever_mode_ratios(max(mode_numbers))
    ideal = np.array([ratios[m - 1] for m in mode_numbers]) * f1
    return float(np.sqrt(np.mean(((frequencies - ideal) / ideal) ** 2)))


def youngs_modulus_from_modes(
    frequencies: np.ndarray,
    mode_numbers: list[int],
    geometry: BeamGeometry,
    *,
    density_kg_m3: float,
) -> float:
    """Estimate the Young modulus using the fundamental implied by all modes."""

    f1 = fundamental_from_modes(frequencies, mode_numbers)
    return young_modulus_from_frequency(geometry, frequency_hz=f1, density_kg_m3=density_kg_m3)
