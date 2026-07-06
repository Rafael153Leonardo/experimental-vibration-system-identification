import numpy as np
from scipy.integrate import odeint

from vibration_id.backbone import (
    analyze_ringdown,
    compare_damping_models,
    fit_backbone,
)
from vibration_id.global_fit import duffing_rhs


def _simulate(omega0_sq, beta, gamma, eta, *, fs=500.0, duration=24.0, x0=6.0):
    t = np.arange(0.0, duration, 1.0 / fs)
    sol = odeint(duffing_rhs, [x0, 0.0], t, args=(omega0_sq, beta, gamma, eta))
    return t, sol[:, 0]


def _f0(omega0_sq: float) -> float:
    return float(np.sqrt(omega0_sq) / (2.0 * np.pi))


def test_backbone_flat_for_linear_stiffness():
    omega0_sq = 400.0
    t, x = _simulate(omega0_sq, beta=0.0, gamma=0.3, eta=0.0)
    fit, _, _ = fit_backbone(t, x, frequency_hz=_f0(omega0_sq))
    # No stiffness nonlinearity -> the backbone barely moves across the decay.
    assert abs(fit.fractional_pull) < 0.01


def test_backbone_detects_hardening_stiffness():
    omega0_sq = 400.0
    t, x = _simulate(omega0_sq, beta=6.0, gamma=0.3, eta=0.0)
    fit, _, _ = fit_backbone(t, x, frequency_hz=_f0(omega0_sq))
    # Hardening Duffing (beta > 0): frequency rises with amplitude, so f vs A^2
    # has a positive slope and a clearly non-negligible pull.
    assert fit.slope_hz_per_amp2 > 0.0
    assert fit.fractional_pull > 0.01


def test_damping_prefers_nonlinear_when_eta_positive():
    omega0_sq = 400.0
    t, x = _simulate(omega0_sq, beta=0.0, gamma=0.2, eta=0.08)
    comparison, *_ = compare_damping_models(t, x, frequency_hz=_f0(omega0_sq))
    assert comparison.prefers_nonlinear
    assert comparison.r2_nonlinear >= comparison.r2_linear


def test_pure_viscous_decay_is_well_fit_by_linear_model():
    omega0_sq = 400.0
    t, x = _simulate(omega0_sq, beta=0.0, gamma=0.3, eta=0.0)
    comparison, *_ = compare_damping_models(t, x, frequency_hz=_f0(omega0_sq))
    # A single exponential should already explain a pure-viscous envelope well.
    assert comparison.r2_linear > 0.99
    assert comparison.q_linear > 0.0


def test_analyze_ringdown_bundles_arrays():
    omega0_sq = 400.0
    t, x = _simulate(omega0_sq, beta=0.0, gamma=0.25, eta=0.05)
    result = analyze_ringdown(t, x, frequency_hz=_f0(omega0_sq))
    assert result.t_window.shape == result.envelope.shape
    assert result.linear_model.shape == result.envelope.shape
    assert result.nonlinear_model.shape == result.envelope.shape
    assert result.amplitude_bins.shape == result.frequency_bins.shape
