import numpy as np

from vibration_id.materials import (
    BeamGeometry,
    natural_frequency_cantilever,
    young_modulus_from_frequency,
)

GEOMETRY = BeamGeometry(length_m=0.27, thickness_m=0.0015, width_m=0.020)


def test_tip_mass_zero_matches_continuous_formula():
    f_continuous = natural_frequency_cantilever(GEOMETRY, young_modulus_pa=200e9, density_kg_m3=7850.0)
    f_lumped_zero = natural_frequency_cantilever(
        GEOMETRY, young_modulus_pa=200e9, density_kg_m3=7850.0, tip_mass_kg=0.0
    )
    assert np.isclose(f_continuous, f_lumped_zero, rtol=1e-9)


def test_tip_mass_lowers_frequency():
    f_no_tip = natural_frequency_cantilever(GEOMETRY, young_modulus_pa=200e9, density_kg_m3=7850.0)
    f_tip = natural_frequency_cantilever(GEOMETRY, young_modulus_pa=200e9, density_kg_m3=7850.0, tip_mass_kg=0.0003)
    assert f_tip < f_no_tip


def test_tip_mass_roundtrip_recovers_modulus():
    young_true = 200e9
    tip = 0.00035
    f = natural_frequency_cantilever(GEOMETRY, young_modulus_pa=young_true, density_kg_m3=7850.0, tip_mass_kg=tip)
    recovered = young_modulus_from_frequency(GEOMETRY, frequency_hz=f, density_kg_m3=7850.0, tip_mass_kg=tip)
    assert np.isclose(recovered, young_true, rtol=1e-9)


def test_ignoring_tip_mass_biases_modulus_low():
    # Steel beam with a real tip mass: ignoring it underestimates E badly.
    young_true = 200e9
    tip = 0.0005
    f = natural_frequency_cantilever(GEOMETRY, young_modulus_pa=young_true, density_kg_m3=7850.0, tip_mass_kg=tip)
    naive = young_modulus_from_frequency(GEOMETRY, frequency_hz=f, density_kg_m3=7850.0)
    assert naive < young_true  # strongly biased low
