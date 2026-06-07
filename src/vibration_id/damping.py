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
    """Estimate Q using the convention Q = f0 / gamma."""

    return float(frequency_hz / gamma)

