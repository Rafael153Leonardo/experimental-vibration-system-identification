"""Ensemble SVD reconstruction across repeated experiments.

Ported and cleaned from the original ``Untitled.ipynb``. Several repetitions of
the same free-vibration experiment are resampled onto a common time axis and
stacked into a data matrix ``X`` of shape ``(M_samples, N_experiments)``. A
truncated SVD then extracts the dominant coherent modes shared across
repetitions and rejects per-trial noise.

This is distinct from the Hankel/HAVOK SVD in :mod:`vibration_id.havok`, which
embeds a *single* signal in delay coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EnsembleSVD:
    common_time: np.ndarray
    data_matrix: np.ndarray  # (M, N), mean-removed per experiment
    mean: np.ndarray  # (N,) removed column means
    U: np.ndarray
    S: np.ndarray
    Vt: np.ndarray

    def energy_fraction(self, rank: int) -> float:
        """Fraction of total spectral energy retained by the first ``rank`` modes."""

        total = float(np.sum(self.S**2))
        if total == 0.0:
            return 0.0
        return float(np.sum(self.S[:rank] ** 2) / total)


def build_ensemble(
    times: list[np.ndarray],
    signals: list[np.ndarray],
    *,
    n_samples: int | None = None,
    remove_mean: bool = True,
) -> EnsembleSVD:
    """Resample experiments onto a common axis and compute the ensemble SVD.

    Each ``(times[j], signals[j])`` pair is one experiment. They are linearly
    interpolated onto the overlapping time window and stacked column-wise.
    """

    from scipy.interpolate import interp1d

    if len(times) != len(signals) or not times:
        raise ValueError("times and signals must be non-empty lists of equal length.")

    t_min = max(float(t[0]) for t in times)
    t_max = min(float(t[-1]) for t in times)
    if t_max <= t_min:
        raise ValueError("Experiments do not share an overlapping time window.")

    m = n_samples or min(len(t) for t in times)
    common = np.linspace(t_min, t_max, m)

    matrix = np.zeros((m, len(signals)), dtype=float)
    for j, (t, x) in enumerate(zip(times, signals)):
        f = interp1d(np.asarray(t, dtype=float), np.asarray(x, dtype=float), kind="linear", fill_value="extrapolate")
        matrix[:, j] = f(common)

    mean = np.mean(matrix, axis=0)
    if remove_mean:
        matrix = matrix - mean

    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    return EnsembleSVD(common_time=common, data_matrix=matrix, mean=mean, U=U, S=S, Vt=Vt)


def reconstruct_rank(ensemble: EnsembleSVD, rank: int) -> tuple[np.ndarray, float]:
    """Rank-``r`` reconstruction of the data matrix and its relative error.

    Returns ``(X_r, relative_frobenius_error)``.
    """

    rank = int(max(1, min(rank, len(ensemble.S))))
    x_r = ensemble.U[:, :rank] @ np.diag(ensemble.S[:rank]) @ ensemble.Vt[:rank, :]
    x = ensemble.data_matrix
    denom = np.linalg.norm(x, "fro")
    error = float(np.linalg.norm(x - x_r, "fro") / denom) if denom else 0.0
    return x_r, error
