"""Linear-model residual analysis for the sensor nonlinearity.

Ported and cleaned from the residual analysis in the original ``completo.py``
(narrative section 4), with the original ``DT=1000`` bug fixed (the savgol
``delta`` must be the sampling period in seconds, not 1000).

The idea: identify the dominant *linear* oscillator from the measured state and
subtract it. The remaining dynamics residual

    r(t) = v'(t) - (a1 q + a2 v),     a1 ~ -omega0^2,  a2 ~ -gamma

is the part of the acceleration the linear model cannot explain. Its time
signature, spectrum, phase-space portrait and dependence on the state ``(q, v)``
expose the nonlinear/sensor effects that the linear fit leaves behind.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LinearResidual:
    omega0_sq: float
    gamma: float
    q: np.ndarray
    v: np.ndarray
    v_dot: np.ndarray
    residual: np.ndarray

    @property
    def frequency_hz(self) -> float:
        return float(np.sqrt(max(self.omega0_sq, 0.0)) / (2.0 * np.pi))

    @property
    def residual_energy_fraction(self) -> float:
        """Fraction of the acceleration variance left in the residual."""

        total = float(np.var(self.v_dot))
        return float(np.var(self.residual) / total) if total else 0.0


def linear_dynamics_residual(q: np.ndarray, v: np.ndarray, *, dt: float) -> LinearResidual:
    """Fit a linear oscillator ``v' = a1 q + a2 v`` and return its residual."""

    q = np.asarray(q, dtype=float)
    v = np.asarray(v, dtype=float)
    if q.shape != v.shape:
        raise ValueError("q and v must have the same shape.")

    theta = np.column_stack([q, v])
    v_dot = np.gradient(v, dt)
    coef = np.linalg.lstsq(theta, v_dot, rcond=None)[0]
    residual = v_dot - theta @ coef
    a1, a2 = coef
    return LinearResidual(
        omega0_sq=-float(a1),
        gamma=-float(a2),
        q=q,
        v=v,
        v_dot=v_dot,
        residual=residual,
    )


def fit_residual_surface(q: np.ndarray, v: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """Fit ``r ~ c_q2 q^2 + c_q3 q^3 + c_qv q v + c_v2 v^2`` to the residual.

    These are the leading nonlinear terms (the linear ones are already removed),
    so the fitted surface is the response surface ``r(q, v)`` that the figures
    slice through. Returns ``[c_q2, c_q3, c_qv, c_v2]``.
    """

    q = np.asarray(q, dtype=float)
    v = np.asarray(v, dtype=float)
    residual = np.asarray(residual, dtype=float)
    theta = np.column_stack([q**2, q**3, q * v, v**2])
    return np.linalg.lstsq(theta, residual, rcond=None)[0]


def evaluate_residual_surface(coeffs: np.ndarray, q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Evaluate the fitted residual response surface on a ``(q, v)`` grid."""

    c_q2, c_q3, c_qv, c_v2 = coeffs
    return c_q2 * q**2 + c_q3 * q**3 + c_qv * q * v + c_v2 * v**2
