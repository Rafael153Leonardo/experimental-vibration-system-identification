import numpy as np

from vibration_id.beam_modes import cantilever_mode_ratios
from vibration_id.digital_twin import BeamParams, DampingParams, DigitalTwin, SensorParams
from vibration_id.spectral import dominant_frequency


def test_default_twin_matches_measured_fundamental():
    twin = DigitalTwin()
    # Seeded with the identified steel-ruler parameters -> ~4.98 Hz.
    assert np.isclose(twin.fundamental_hz(), 4.982, atol=0.02)


def test_modal_ladder_follows_beta_squared():
    twin = DigitalTwin()
    modes = twin.natural_frequencies(4)
    ratios = modes / modes[0]
    assert np.allclose(ratios, cantilever_mode_ratios(4), rtol=1e-6)


def test_tip_mass_lowers_frequency_monotonically():
    twin = DigitalTwin()
    masses = np.array([0.0, 0.5e-3, 1.0e-3, 2.0e-3])
    freqs = twin.tip_mass_frequency_shift(masses)
    assert np.all(np.diff(freqs) < 0.0)


def test_thinner_blade_reads_as_softer_material():
    thick = DigitalTwin(beam=BeamParams(thickness_m=1.5e-3))
    thin = DigitalTwin(beam=BeamParams(thickness_m=0.55e-3))
    # Same measured frequency would be read as a lower modulus for a thicker
    # assumed blade (E ∝ 1/h^2), which is exactly the paper's trap.
    assert thick.fundamental_hz() > thin.fundamental_hz()


def test_material_verdict_is_steel_for_default():
    twin = DigitalTwin()
    assert "steel" in twin.material_verdict().lower()
    assert np.isclose(twin.young_modulus_estimate() / 1e9, 205.3, atol=5.0)


def test_free_decay_signal_recovers_fundamental():
    twin = DigitalTwin()
    fd = twin.simulate_free_decay(amplitude_mm=20.0, duration_s=15.0, fs=1000.0)
    f0 = dominant_frequency(fd.t, fd.signal, fmin=1.0, fmax=80.0)
    assert np.isclose(f0, twin.fundamental_hz(), atol=0.05)


def test_frequency_response_peaks_at_modes():
    twin = DigitalTwin()
    modes = twin.natural_frequencies(4)
    freqs = np.linspace(2.0, 200.0, 8000)
    mag, _ = twin.frequency_response(freqs, n_modes=4)
    for f_n in modes:
        band = (freqs > f_n - 1.0) & (freqs < f_n + 1.0)
        peak_f = freqs[band][np.argmax(mag[band])]
        assert abs(peak_f - f_n) < 0.5


def test_sensor_cubic_injects_third_harmonic():
    linear = DigitalTwin(sensor=SensorParams(cubic=0.0))
    cubic = DigitalTwin(sensor=SensorParams(cubic=3e-3))
    q = np.linspace(-10.0, 10.0, 100)
    # A pure gain leaves a straight line; the cubic bends it.
    assert np.allclose(linear.sensor.apply(q), q)
    assert not np.allclose(cubic.sensor.apply(q), q)


def test_stronger_damping_lowers_quality_factor():
    soft = DigitalTwin(damping=DampingParams(gamma=0.10, eta=0.0))
    hard = DigitalTwin(damping=DampingParams(gamma=0.40, eta=0.0))
    q_soft = 2.0 * np.pi * soft.fundamental_hz() / soft.damping.gamma
    q_hard = 2.0 * np.pi * hard.fundamental_hz() / hard.damping.gamma
    assert q_soft > q_hard
