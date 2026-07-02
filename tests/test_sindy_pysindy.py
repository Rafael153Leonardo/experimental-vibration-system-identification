"""Regression tests for the optional pysindy integration.

These run only when pysindy is installed (requirements-advanced.txt); the CI
advanced job exercises them. They guard the pysindy>=2 fit API used in
``fit_linear_sindy``, which broke silently under pysindy 1.x + NumPy 2.
"""

import numpy as np
import pytest
from scipy.integrate import odeint

from vibration_id.sindy_model import fit_linear_sindy, physical_params_from_sindy, state_from_position

pytest.importorskip("pysindy")


def _damped_oscillator(k: float, c: float, dt: float, duration: float) -> np.ndarray:
    t = np.arange(0.0, duration, dt)
    return t, odeint(lambda s, _t: [s[1], -k * s[0] - c * s[1]], [1.0, 0.0], t)


def test_fit_linear_sindy_recovers_oscillator_parameters():
    k, c = (2.0 * np.pi * 3.0) ** 2, 0.4
    dt = 0.002
    _, state = _damped_oscillator(k, c, dt, duration=8.0)

    model = fit_linear_sindy([state], dt=dt)
    params = physical_params_from_sindy(model)

    assert params.frequency_hz == pytest.approx(3.0, rel=0.02)
    assert params.stiffness_over_mass == pytest.approx(k, rel=0.05)
    assert params.damping_over_mass == pytest.approx(c, rel=0.10)


def test_fit_linear_sindy_from_position_only():
    k, c = (2.0 * np.pi * 3.0) ** 2, 0.4
    dt = 0.002
    t, state = _damped_oscillator(k, c, dt, duration=8.0)

    rebuilt = state_from_position(t, state[:, 0], method="savgol", window_length=21, polyorder=3)
    model = fit_linear_sindy([rebuilt], dt=dt)
    params = physical_params_from_sindy(model)

    assert params.frequency_hz == pytest.approx(3.0, rel=0.05)
