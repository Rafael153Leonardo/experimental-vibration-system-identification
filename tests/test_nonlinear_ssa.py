import numpy as np
from scipy.integrate import odeint

from vibration_id.nonlinear_id import (
    identify_duffing,
    identify_sensor_map,
    simulate_model,
)
from vibration_id.ssa import build_ensemble, reconstruct_rank


def _simulate_duffing(omega0_sq, gamma, alpha, *, dt=0.001, n=4000, q0=4.0):
    t = np.arange(n) * dt

    def rhs(state, _t):
        q, v = state
        return [v, -omega0_sq * q - gamma * v - alpha * q**3]

    sol = odeint(rhs, [q0, 0.0], t)
    return t, sol[:, 0], sol[:, 1]


def test_identify_duffing_recovers_parameters():
    omega0_sq, gamma, alpha = 13900.0, 0.36, 6.9
    t, q, v = _simulate_duffing(omega0_sq, gamma, alpha)

    params = identify_duffing(q, v, dt=float(t[1] - t[0]))

    assert np.isclose(params.omega0_sq, omega0_sq, rtol=0.05)
    assert np.isclose(params.gamma, gamma, rtol=0.15)
    assert np.isclose(params.alpha, alpha, rtol=0.15)


def test_sensor_output_map_recovers_polynomial():
    q = np.linspace(-5.0, 5.0, 500)
    true = np.array([0.2, 1.5, -0.05, 0.01])
    measured = true[0] + true[1] * q + true[2] * q**2 + true[3] * q**3

    sensor = identify_sensor_map(q, measured)

    assert np.allclose(sensor.coefficients, true, atol=1e-6)
    assert sensor.nonlinearity_strength > 0.0


def test_simulate_model_runs_and_applies_sensor():
    omega0_sq, gamma, alpha = 13900.0, 0.36, 6.9
    t, q, v = _simulate_duffing(omega0_sq, gamma, alpha, n=2000)
    params = identify_duffing(q, v, dt=float(t[1] - t[0]))
    sensor = identify_sensor_map(q, q)  # identity-ish map

    q_sim, y_sim = simulate_model(params, sensor, q0=q[0], v0=v[0], t=t)

    assert q_sim.shape == t.shape
    assert y_sim.shape == t.shape


def test_ensemble_svd_rank_reconstruction_error_decreases():
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 2.0, 800)
    base = np.exp(-0.5 * t) * np.cos(2.0 * np.pi * 5.0 * t)
    times = [t for _ in range(8)]
    signals = [base + rng.normal(0.0, 0.05, size=t.size) for _ in range(8)]

    ens = build_ensemble(times, signals)
    _, err1 = reconstruct_rank(ens, 1)
    _, err3 = reconstruct_rank(ens, 3)

    assert 0.0 <= err3 <= err1
    assert ens.energy_fraction(1) > 0.5
