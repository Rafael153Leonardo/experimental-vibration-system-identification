import numpy as np

from vibration_id.damping import fit_exponential_envelope
from vibration_id.havok import havok_svd, identify_havok
from vibration_id.pinn import pinn_available
from vibration_id.sindy_model import physical_params_from_sindy
from vibration_id.spectral import dominant_frequency
from vibration_id.synthetic import damped_oscillator


def test_synthetic_frequency_and_damping_are_recovered():
    t, x = damped_oscillator(duration=10.0, frequency_hz=12.5, gamma=0.4, noise_std=0.0)

    f0 = dominant_frequency(t, x, fmin=1.0, fmax=30.0)
    fit = fit_exponential_envelope(t, x, tmin=0.0, tmax=5.0)

    assert np.isclose(f0, 12.5, atol=0.2)
    assert np.isclose(fit.gamma, 0.4, atol=0.03)


def test_havok_shapes_are_consistent():
    t, x = damped_oscillator(duration=2.0, frequency_hz=8.0, gamma=0.2, noise_std=0.0)

    _, _, s, vt = havok_svd(x, delays=40)
    a, b, z, u = identify_havok(t, s, vt, rank=5)

    assert a.shape == (4, 4)
    assert b.shape == (4, 1)
    assert z.shape[0] == 4
    assert u.shape[0] == 1


def test_sindy_parameter_extraction_with_feature_names():
    omega = 2.0 * np.pi * 10.0
    zeta = 0.05

    class FakeSindyModel:
        def coefficients(self):
            return np.array(
                [
                    [0.0, 1.0, 0.0],
                    [0.0, -(omega**2), -(2.0 * zeta * omega)],
                ]
            )

        def get_feature_names(self):
            return ["1", "x", "v"]

    params = physical_params_from_sindy(FakeSindyModel())

    assert np.isclose(params.frequency_hz, 10.0, atol=1e-9)
    assert np.isclose(params.zeta, zeta, atol=1e-9)


def test_pinn_module_reports_availability():
    assert isinstance(pinn_available(), bool)
