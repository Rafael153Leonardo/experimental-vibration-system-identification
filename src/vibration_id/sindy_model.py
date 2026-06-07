from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SindyPhysicalParams:
    omega_n: float
    frequency_hz: float
    zeta: float
    stiffness_over_mass: float
    damping_over_mass: float


def state_from_position(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Build [x, v] state from position and a time vector."""

    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    dt = float(np.median(np.diff(t)))
    v = np.gradient(x, dt, edge_order=2)
    return np.column_stack([x, v])


def fit_linear_sindy(
    trajectories: list[np.ndarray],
    *,
    dt: float,
    threshold: float = 0.01,
    alpha: float = 0.05,
):
    """Fit a linear SINDy oscillator model to multiple [x, v] trajectories."""

    try:
        import pysindy as ps
    except ImportError as exc:
        raise ImportError("Install pysindy to use fit_linear_sindy.") from exc

    model = ps.SINDy(
        optimizer=ps.STLSQ(threshold=threshold, alpha=alpha),
        feature_library=ps.PolynomialLibrary(degree=1),
        feature_names=["x", "v"],
    )
    model.fit(trajectories, t=dt, multiple_trajectories=True)
    return model


def physical_params_from_sindy(model) -> SindyPhysicalParams:
    """Extract oscillator parameters from a fitted linear SINDy model."""

    coefs = np.asarray(model.coefficients(), dtype=float)
    k_over_m = abs(coefs[1, 1])
    c_over_m = abs(coefs[1, 2])
    omega_n = float(np.sqrt(k_over_m))
    zeta = float(c_over_m / (2.0 * omega_n)) if omega_n else np.nan
    return SindyPhysicalParams(
        omega_n=omega_n,
        frequency_hz=omega_n / (2.0 * np.pi),
        zeta=zeta,
        stiffness_over_mass=float(k_over_m),
        damping_over_mass=float(c_over_m),
    )

