"""Unit tests for the PINN helpers.

These guard two regressions that once made the PINN demo converge to a
physically wrong frequency (1.84 Hz on an 18.7 Hz signal):

- ``_inverse_softplus`` overflowed to inf for realistic stiffness values, and
- the unconstrained strided subsample aliased the signal, so the FFT-based
  stiffness initialization locked onto an alias.
"""

import numpy as np
import pytest

from vibration_id.pinn import _inverse_softplus, _subsample, pinn_available


def test_inverse_softplus_large_values_stay_finite():
    assert _inverse_softplus(13805.0) == pytest.approx(13805.0)
    assert np.isfinite(_inverse_softplus(1e8))


def test_inverse_softplus_small_values_invert_softplus():
    raw = _inverse_softplus(5.0)
    assert np.log1p(np.exp(raw)) == pytest.approx(5.0)


def test_subsample_respects_min_sampling_rate():
    fs, f_signal = 1000.0, 18.7
    t = np.arange(45000) / fs
    x = np.sin(2.0 * np.pi * f_signal * t)

    t_sub, x_sub = _subsample(t, x, 900, min_sampling_hz=8.0 * f_signal)

    assert len(t_sub) <= 900
    assert len(t_sub) == len(x_sub)
    dt = float(np.median(np.diff(t_sub)))
    assert 1.0 / dt >= 8.0 * f_signal - 1e-6


def test_subsample_returns_short_input_unchanged():
    t = np.arange(100) / 100.0
    x = np.ones_like(t)

    t_sub, x_sub = _subsample(t, x, 900, min_sampling_hz=50.0)

    assert len(t_sub) == 100
    assert np.array_equal(x_sub, x)


def test_build_model_accepts_large_stiffness():
    if not pinn_available():
        pytest.skip("torch not installed")
    import torch

    from vibration_id.pinn import build_pinn_model

    model = build_pinn_model(t_mean=0.0, t_scale=1.0, initial_stiffness_over_mass=13805.0)
    stiffness = model.stiffness_over_mass.detach()
    assert torch.isfinite(stiffness).all()
    assert float(stiffness) == pytest.approx(13805.0, rel=1e-3)
