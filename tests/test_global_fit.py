import numpy as np
from scipy.integrate import odeint

from vibration_id.damping import hilbert_envelope
from vibration_id.global_fit import (
    duffing_rhs,
    fit_global_duffing,
    fit_nonlinear_envelope,
    nonlinear_envelope,
    simulate_duffing,
)


def _simulate(omega0_sq, beta, gamma, eta, *, fs=500.0, duration=10.0, x0=4.0):
    t = np.arange(0.0, duration, 1.0 / fs)
    sol = odeint(duffing_rhs, [x0, 0.0], t, args=(omega0_sq, beta, gamma, eta))
    return t, sol[:, 0]


def test_nonlinear_envelope_reduces_to_exponential_when_eta_zero():
    t = np.linspace(0.0, 5.0, 200)
    r = nonlinear_envelope(t, r0=4.0, gamma=0.4, eta=0.0)
    assert np.allclose(r, 4.0 * np.exp(-0.4 * t / 2.0), atol=1e-9)


def test_envelope_fit_recovers_damping_parameters():
    gamma, eta = 0.3, 0.1
    t, x = _simulate(400.0, 0.0, gamma, eta)
    env = hilbert_envelope(x)

    fit = fit_nonlinear_envelope(t, env)

    assert np.isclose(fit.gamma, gamma, rtol=0.25)
    assert np.isclose(fit.eta, eta, rtol=0.4)
    assert fit.r0 > 0


def test_global_duffing_recovers_frequency():
    omega0_sq, beta, gamma, eta = 400.0, 5.0, 0.3, 0.05
    t, x = _simulate(omega0_sq, beta, gamma, eta)

    fit = fit_global_duffing(
        t, x, gamma=gamma, eta=eta, omega0_sq_guess=380.0, beta_guess=0.0, fit_seconds=8.0
    )

    assert np.isclose(fit.omega0_sq, omega0_sq, rtol=0.05)
    assert fit.mse < 1e-2
    x_sim = simulate_duffing(fit, t)
    assert x_sim.shape == t.shape
