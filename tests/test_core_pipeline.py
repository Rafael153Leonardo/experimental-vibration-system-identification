import numpy as np

from vibration_id.damping import fit_exponential_envelope
from vibration_id.havok import havok_svd, identify_havok
from vibration_id.materials import (
    BeamGeometry,
    natural_frequency_cantilever,
    rank_materials_by_frequency,
    rank_materials_by_young,
    young_modulus_from_frequency,
)
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


def test_euler_bernoulli_material_estimate_roundtrip():
    geometry = BeamGeometry(length_m=0.30, thickness_m=0.00233, width_m=0.025)
    young_pa = 3.2e9
    density = 1050.0

    frequency = natural_frequency_cantilever(
        geometry,
        young_modulus_pa=young_pa,
        density_kg_m3=density,
    )
    recovered_young = young_modulus_from_frequency(
        geometry,
        frequency_hz=frequency,
        density_kg_m3=density,
    )

    assert np.isclose(recovered_young, young_pa, rtol=1e-12)

    by_young = rank_materials_by_young(recovered_young)
    assert by_young[0].candidate.name in {"Acrilico", "Poliestireno"}

    by_frequency = rank_materials_by_frequency(geometry, frequency_hz=frequency)
    assert by_frequency[0].candidate.name == "Poliestireno"
