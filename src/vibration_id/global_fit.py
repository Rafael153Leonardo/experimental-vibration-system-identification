"""Global Duffing fit with nonlinear (amplitude-dependent) damping.

Ported and cleaned from the original ``prof/definitivo.ipynb`` (narrative
section 11). The full model is a free Duffing oscillator with mixed linear and
amplitude-dependent damping::

    x'' = -omega0_sq * x - beta * x**3 - (gamma + eta * x**2) * x'

It is identified in three stages, exactly as in the original workflow:

1. **Nonlinear envelope fit** -- fit the analytic amplitude law to the Hilbert
   envelope to recover the linear damping ``gamma`` and the nonlinear damping
   ``eta``.
2. **Phase/frequency calibration** -- fit a damped cosine to the early signal to
   recover a precise ``omega`` and phase ``phi`` (hence the initial conditions).
3. **Global Duffing optimization** -- with ``gamma`` and ``eta`` fixed, optimize
   ``(omega0_sq, beta, phi)`` by minimizing the mean-squared error between the
   integrated model and the measured signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EnvelopeFit:
    r0: float
    gamma: float
    eta: float


@dataclass(frozen=True)
class GlobalDuffingFit:
    omega0_sq: float
    beta: float
    gamma: float
    eta: float
    phi: float
    x0: float
    v0: float
    mse: float

    @property
    def frequency_hz(self) -> float:
        return float(np.sqrt(max(self.omega0_sq, 0.0)) / (2.0 * np.pi))


def nonlinear_envelope(t: np.ndarray, r0: float, gamma: float, eta: float) -> np.ndarray:
    """Analytic amplitude law ``r(t)`` for linear + nonlinear damping.

    ``r(t) = sqrt( exp(-gamma t) / (1/r0^2 + (eta/(4 gamma)) (1 - exp(-gamma t))) )``.
    Reduces to a pure exponential ``r0 exp(-gamma t / 2)`` as ``eta -> 0``.
    """

    t = np.asarray(t, dtype=float)
    gamma = max(float(gamma), 1e-9)
    numerator = np.exp(-gamma * t)
    nonlinear_term = (eta / (4.0 * gamma)) * (1.0 - np.exp(-gamma * t))
    denominator = (1.0 / (r0**2)) + nonlinear_term
    return np.sqrt(numerator / denominator)


def fit_nonlinear_envelope(
    t: np.ndarray,
    envelope: np.ndarray,
    *,
    smooth_window: int = 51,
    polyorder: int = 3,
) -> EnvelopeFit:
    """Fit :func:`nonlinear_envelope` to a measured (Hilbert) envelope."""

    from scipy.optimize import curve_fit
    from scipy.signal import savgol_filter

    t = np.asarray(t, dtype=float)
    envelope = np.asarray(envelope, dtype=float)
    if smooth_window and len(envelope) > smooth_window:
        win = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
        envelope = savgol_filter(envelope, win, polyorder)

    r0_guess = float(np.max(envelope))
    # curve_fit parameter order matches nonlinear_envelope(t, r0, gamma, eta)
    p0 = [r0_guess, 0.3, 0.05]
    bounds = ([1e-9, 0.0, 0.0], [np.inf, np.inf, np.inf])
    popt, _ = curve_fit(nonlinear_envelope, t, envelope, p0=p0, bounds=bounds, maxfev=20000)
    r0, gamma, eta = popt
    return EnvelopeFit(r0=float(r0), gamma=float(gamma), eta=float(eta))


def damped_cosine(t: np.ndarray, amplitude: float, gamma: float, omega: float, phi: float) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    return amplitude * np.exp(-gamma * t) * np.cos(omega * t + phi)


def calibrate_phase(
    t: np.ndarray,
    x: np.ndarray,
    *,
    omega_guess: float,
    gamma_guess: float = 0.35,
    fit_seconds: float = 2.0,
) -> tuple[float, float, float, float]:
    """Fit a damped cosine to the early signal; return ``(A, gamma, omega, phi)``."""

    from scipy.optimize import curve_fit

    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    mask = t < (t[0] + fit_seconds)
    if np.count_nonzero(mask) < 10:
        mask = np.ones_like(t, dtype=bool)

    p0 = [float(np.max(np.abs(x))), gamma_guess, omega_guess, 0.0]
    popt, _ = curve_fit(damped_cosine, t[mask], x[mask], p0=p0, maxfev=20000)
    amplitude, gamma, omega, phi = popt
    return float(amplitude), float(gamma), float(abs(omega)), float(phi)


def duffing_rhs(state: np.ndarray, _t: float, omega0_sq: float, beta: float, gamma: float, eta: float) -> list[float]:
    x, v = state
    return [v, -omega0_sq * x - beta * x**3 - (gamma + eta * x**2) * v]


def _decimate(t: np.ndarray, x: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or len(t) <= max_points:
        return t, x
    idx = np.linspace(0, len(t) - 1, max_points).astype(int)
    return t[idx], x[idx]


def fit_global_duffing(
    t: np.ndarray,
    x: np.ndarray,
    *,
    gamma: float,
    eta: float,
    omega0_sq_guess: float,
    beta_guess: float = 0.0,
    fit_seconds: float = 5.0,
    max_points: int = 2000,
    calibrate: bool = True,
) -> GlobalDuffingFit:
    """Optimize ``(omega0_sq, beta, phi)`` to match the measured signal.

    ``gamma`` and ``eta`` are held fixed (from the envelope stage). The initial
    conditions are derived from the optimized phase, following the original
    workflow: ``x0 = A cos(phi)``, ``v0 = -A sqrt(omega0_sq) sin(phi)``.

    Direct MSE over a long oscillatory record is multimodal in frequency, so by
    default a damped-cosine calibration (the original stage 2) first locks a
    precise ``omega`` and phase seed before the Nelder-Mead refinement.
    """

    from scipy.integrate import odeint
    from scipy.optimize import minimize

    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    mask = t < (t[0] + fit_seconds)
    t_fit, x_fit = _decimate(t[mask] - t[0], x[mask], max_points)
    amplitude = float(np.max(np.abs(x_fit)))

    omega_sq_seed = float(omega0_sq_guess)
    phi_seed = 0.0
    if calibrate:
        try:
            _, _, omega_c, phi_c = calibrate_phase(
                t_fit, x_fit, omega_guess=np.sqrt(abs(omega0_sq_guess)), gamma_guess=gamma, fit_seconds=fit_seconds
            )
            omega_sq_seed = omega_c**2
            phi_seed = phi_c
        except Exception:
            pass

    def cost(params: np.ndarray) -> float:
        omega0_sq, beta, phi = params
        x0 = amplitude * np.cos(phi)
        v0 = -amplitude * np.sqrt(abs(omega0_sq)) * np.sin(phi)
        sol = odeint(duffing_rhs, [x0, v0], t_fit, args=(omega0_sq, beta, gamma, eta))
        return float(np.mean((x_fit - sol[:, 0]) ** 2))

    res = minimize(cost, [omega_sq_seed, beta_guess, phi_seed], method="Nelder-Mead", tol=1e-4)
    omega0_sq, beta, phi = res.x
    x0 = amplitude * np.cos(phi)
    v0 = -amplitude * np.sqrt(abs(omega0_sq)) * np.sin(phi)
    return GlobalDuffingFit(
        omega0_sq=float(omega0_sq),
        beta=float(beta),
        gamma=float(gamma),
        eta=float(eta),
        phi=float(phi),
        x0=float(x0),
        v0=float(v0),
        mse=float(res.fun),
    )


def simulate_duffing(fit: GlobalDuffingFit, t: np.ndarray) -> np.ndarray:
    """Integrate the identified global model over ``t`` (reset to start at 0)."""

    from scipy.integrate import odeint

    t = np.asarray(t, dtype=float)
    t0 = t - t[0]
    sol = odeint(duffing_rhs, [fit.x0, fit.v0], t0, args=(fit.omega0_sq, fit.beta, fit.gamma, fit.eta))
    return sol[:, 0]
