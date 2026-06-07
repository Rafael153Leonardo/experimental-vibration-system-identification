import numpy as np

from vibration_id.damping import fit_exponential_envelope
from vibration_id.spectral import dominant_frequency
from vibration_id.synthetic import damped_oscillator


def test_synthetic_frequency_and_damping_are_recovered():
    t, x = damped_oscillator(duration=10.0, frequency_hz=12.5, gamma=0.4, noise_std=0.0)

    f0 = dominant_frequency(t, x, fmin=1.0, fmax=30.0)
    fit = fit_exponential_envelope(t, x, tmin=0.0, tmax=5.0)

    assert np.isclose(f0, 12.5, atol=0.2)
    assert np.isclose(fit.gamma, 0.4, atol=0.03)

