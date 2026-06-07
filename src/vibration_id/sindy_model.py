from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vibration_id.preprocessing import velocity_savgol


@dataclass(frozen=True)
class SindyPhysicalParams:
    omega_n: float
    frequency_hz: float
    zeta: float
    stiffness_over_mass: float
    damping_over_mass: float


def state_from_position(
    t: np.ndarray,
    x: np.ndarray,
    *,
    method: str = "gradient",
    window_length: int = 81,
    polyorder: int = 3,
) -> np.ndarray:
    """Build [x, v] state from position and a time vector."""

    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    dt = float(np.median(np.diff(t)))
    if method == "gradient":
        v = np.gradient(x, dt, edge_order=2)
    elif method == "savgol":
        v = velocity_savgol(t, x, window_length=window_length, polyorder=polyorder)
    else:
        raise ValueError("method must be 'gradient' or 'savgol'")
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
        feature_library=ps.PolynomialLibrary(degree=1, include_bias=True),
        feature_names=["x", "v"],
    )
    model.fit(trajectories, t=dt, multiple_trajectories=True)
    return model


def physical_params_from_sindy(model) -> SindyPhysicalParams:
    """Extract oscillator parameters from a fitted linear SINDy model."""

    coefs = np.asarray(model.coefficients(), dtype=float)
    try:
        names = list(model.get_feature_names())
        x_idx = names.index("x")
        v_idx = names.index("v")
    except (AttributeError, ValueError):
        x_idx = 1 if coefs.shape[1] >= 3 else 0
        v_idx = 2 if coefs.shape[1] >= 3 else 1

    k_over_m = abs(coefs[1, x_idx])
    c_over_m = abs(coefs[1, v_idx])
    omega_n = float(np.sqrt(k_over_m))
    zeta = float(c_over_m / (2.0 * omega_n)) if omega_n else np.nan
    return SindyPhysicalParams(
        omega_n=omega_n,
        frequency_hz=omega_n / (2.0 * np.pi),
        zeta=zeta,
        stiffness_over_mass=float(k_over_m),
        damping_over_mass=float(c_over_m),
    )
