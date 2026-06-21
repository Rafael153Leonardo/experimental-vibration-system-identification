import numpy as np
from scipy.integrate import odeint

from vibration_id.sensor_residual import (
    evaluate_residual_surface,
    fit_residual_surface,
    linear_dynamics_residual,
)


def _simulate(omega0_sq, beta, gamma, eta, *, fs=500.0, duration=8.0, x0=4.0):
    t = np.arange(0.0, duration, 1.0 / fs)

    def rhs(state, _t):
        q, v = state
        return [v, -omega0_sq * q - beta * q**3 - (gamma + eta * q**2) * v]

    sol = odeint(rhs, [x0, 0.0], t)
    return t, sol[:, 0], sol[:, 1]


def test_pure_linear_oscillator_has_negligible_residual():
    t, q, v = _simulate(400.0, 0.0, 0.3, 0.0)
    res = linear_dynamics_residual(q, v, dt=float(t[1] - t[0]))

    assert np.isclose(res.omega0_sq, 400.0, rtol=0.05)
    assert np.isclose(res.gamma, 0.3, rtol=0.1)
    assert res.residual_energy_fraction < 1e-3


def test_nonlinear_oscillator_leaves_structured_residual():
    omega0_sq, beta, gamma, eta = 400.0, 8.0, 0.3, 0.1
    t, q, v = _simulate(omega0_sq, beta, gamma, eta)
    res = linear_dynamics_residual(q, v, dt=float(t[1] - t[0]))

    # The best linear fit is biased high by the cubic stiffness (finite-amplitude
    # stiffening), so we only require it stays in a sane range, not exact.
    assert omega0_sq * 0.8 < res.omega0_sq < omega0_sq * 1.5
    assert res.residual_energy_fraction > 1e-3

    # The residual surface should pick up the cubic stiffness (r ~ -beta q^3).
    coeffs = fit_residual_surface(res.q, res.v, res.residual)
    assert coeffs[1] < 0  # c_q3 negative, matching -beta q^3 with beta > 0

    recon = evaluate_residual_surface(coeffs, res.q, res.v)
    assert recon.shape == res.residual.shape
