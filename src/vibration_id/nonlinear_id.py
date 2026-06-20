"""Nonlinear oscillator identification with a separate sensor output map.

Ported and cleaned from the original ``sss.ipynb`` workflow. The key idea is to
separate two effects that the raw sensor signal mixes together:

- the **mechanical dynamics** of the beam, modelled as a free Duffing
  oscillator ``v' = -omega0^2 q - gamma v - alpha q^3``;
- the **static sensor nonlinearity**, modelled as a polynomial output map
  ``h(q) = a0 + a1 q + a2 q^2 + a3 q^3`` mapping the true displacement to the
  measured signal.

The original notebook used a hand-written sequentially-thresholded least squares
(STLSQ) routine; it is reproduced here without external dependencies so the
identification runs with only numpy/scipy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DuffingParams:
    """Identified free-Duffing parameters ``v' = -omega0_sq q - gamma v - alpha q^3``."""

    omega0_sq: float
    gamma: float
    alpha: float

    @property
    def frequency_hz(self) -> float:
        return float(np.sqrt(max(self.omega0_sq, 0.0)) / (2.0 * np.pi))


@dataclass(frozen=True)
class SensorOutputMap:
    """Polynomial sensor map ``h(q) = a0 + a1 q + a2 q^2 + a3 q^3``."""

    coefficients: np.ndarray  # [a0, a1, a2, a3]

    def __call__(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        a0, a1, a2, a3 = self.coefficients
        return a0 + a1 * q + a2 * q**2 + a3 * q**3

    @property
    def nonlinearity_strength(self) -> float:
        """Relative weight of the nonlinear terms vs. the linear gain."""

        a0, a1, a2, a3 = self.coefficients
        linear = abs(a1)
        nonlinear = abs(a2) + abs(a3)
        return float(nonlinear / linear) if linear else float("inf")


def stlsq(theta: np.ndarray, target: np.ndarray, *, threshold: float = 1e-3, max_iter: int = 50) -> np.ndarray:
    """Sequentially-thresholded least squares (the SINDy sparsity routine).

    Reproduces the hand-written ``STLSQ`` from the original ``sss.ipynb`` so the
    identification has no hard dependency on ``pysindy``.
    """

    theta = np.asarray(theta, dtype=float)
    target = np.asarray(target, dtype=float)
    xi = np.linalg.lstsq(theta, target, rcond=None)[0]
    for _ in range(max_iter):
        small = np.abs(xi) < threshold
        if np.all(small):
            break
        xi[small] = 0.0
        big = ~small
        xi[big] = np.linalg.lstsq(theta[:, big], target, rcond=None)[0]
    return xi


def identify_duffing(
    q: np.ndarray,
    v: np.ndarray,
    *,
    dt: float,
    threshold: float = 1e-3,
) -> DuffingParams:
    """Identify the free-Duffing dynamics from displacement and velocity.

    Fits ``v_dot = -omega0_sq q - gamma v - alpha q^3`` via STLSQ over the
    restricted physical library ``[q, v, q^3]``.
    """

    q = np.asarray(q, dtype=float)
    v = np.asarray(v, dtype=float)
    if q.shape != v.shape:
        raise ValueError("q and v must have the same shape.")

    theta = np.column_stack([q, v, q**3])
    v_dot = np.gradient(v, dt)
    xi = stlsq(theta, v_dot, threshold=threshold)
    return DuffingParams(omega0_sq=-float(xi[0]), gamma=-float(xi[1]), alpha=-float(xi[2]))


def identify_sensor_map(q: np.ndarray, measured: np.ndarray) -> SensorOutputMap:
    """Fit the static polynomial sensor map ``h(q)`` by least squares."""

    q = np.asarray(q, dtype=float)
    measured = np.asarray(measured, dtype=float)
    theta = np.column_stack([np.ones_like(q), q, q**2, q**3])
    coeffs = np.linalg.lstsq(theta, measured, rcond=None)[0]
    return SensorOutputMap(coefficients=np.asarray(coeffs, dtype=float))


def simulate_model(
    params: DuffingParams,
    sensor: SensorOutputMap,
    *,
    q0: float,
    v0: float,
    t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the Duffing model and apply the sensor map.

    Returns ``(q_sim, y_sim)`` where ``q_sim`` is the true displacement and
    ``y_sim = h(q_sim)`` is the predicted sensor reading.
    """

    from scipy.integrate import odeint

    t = np.asarray(t, dtype=float)

    def dynamics(state: np.ndarray, _t: float) -> list[float]:
        q, v = state
        return [v, -params.omega0_sq * q - params.gamma * v - params.alpha * q**3]

    state = odeint(dynamics, [float(q0), float(v0)], t)
    q_sim = state[:, 0]
    return q_sim, sensor(q_sim)
