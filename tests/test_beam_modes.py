import numpy as np

from vibration_id.beam_modes import (
    cantilever_mode_frequencies,
    cantilever_mode_ratios,
    fundamental_from_modes,
    modal_ratio_deviation,
    youngs_modulus_from_modes,
)
from vibration_id.materials import BeamGeometry


def test_mode_ratios_are_not_integer_harmonics():
    r = cantilever_mode_ratios(4)
    # cantilever overtones follow beta_n^2, not 1:2:3:4
    assert np.allclose(r, [1.0, 6.267, 17.55, 34.39], rtol=1e-2)


def test_fundamental_recovered_from_higher_modes():
    f1_true = 5.0
    freqs = cantilever_mode_frequencies(f1_true, 4)
    # use only modes 2..4, as in the forced-vibration experiment
    f1 = fundamental_from_modes(freqs[1:], [2, 3, 4])
    assert np.isclose(f1, f1_true, rtol=1e-6)


def test_ideal_clamp_has_negligible_ratio_deviation():
    freqs = cantilever_mode_frequencies(5.0, 4)
    assert modal_ratio_deviation(freqs, [1, 2, 3, 4]) < 1e-6


def test_measured_inox_modes_match_ideal_clamp_and_steel():
    # Real forced-mode measurement of the inox ruler.
    measured = np.array([31.2, 86.8, 173.5])  # modes 2, 3, 4
    f1 = fundamental_from_modes(measured, [2, 3, 4])
    assert np.isclose(f1, 4.98, atol=0.1)  # matches the free-vibration fundamental

    dev = modal_ratio_deviation(measured, [2, 3, 4])
    assert dev < 0.02  # within ~2% of the ideal cantilever ladder -> ideal clamp

    geometry = BeamGeometry(length_m=0.30, thickness_m=0.00055, width_m=0.025)
    e_pa = youngs_modulus_from_modes(measured, [2, 3, 4], geometry, density_kg_m3=7850.0)
    assert 180e9 < e_pa < 220e9  # reconciles to bulk steel
