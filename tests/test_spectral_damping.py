import numpy as np

from vibration_id.damping import (
    frequency_to_decay_ratio,
    quality_factor,
    quality_factor_half_power,
)
from vibration_id.spectral import compute_fft, dominant_frequency


def test_windowed_fft_preserves_tone_amplitude():
    fs = 1000.0
    t = np.arange(0, 4.0, 1.0 / fs)
    amplitude = 2.5
    x = amplitude * np.sin(2.0 * np.pi * 17.0 * t)

    freqs, amp = compute_fft(t, x, window="hann")
    peak = amp[np.argmax(amp)]

    # Coherent-gain normalization should recover the true tone amplitude.
    assert np.isclose(peak, amplitude, rtol=0.02)


def test_parabolic_interpolation_refines_off_bin_frequency():
    fs = 1000.0
    t = np.arange(0, 2.0, 1.0 / fs)  # bin spacing 0.5 Hz
    f_true = 17.27  # deliberately between bins
    x = np.sin(2.0 * np.pi * f_true * t)

    f_interp = dominant_frequency(t, x, fmin=1.0, fmax=80.0, interpolate=True)
    f_bin = dominant_frequency(t, x, fmin=1.0, fmax=80.0, interpolate=False)

    assert abs(f_interp - f_true) < abs(f_bin - f_true)
    assert np.isclose(f_interp, f_true, atol=0.1)


def test_quality_factor_is_physical_and_distinct_from_ratio():
    f0 = 18.0
    gamma = 0.4
    assert np.isclose(quality_factor(f0, gamma), 2.0 * np.pi * f0 / gamma)
    assert np.isclose(frequency_to_decay_ratio(f0, gamma), f0 / gamma)


def test_half_power_q_on_lorentzian_peak():
    # Lorentzian centered at f0 with half-width hwhm -> Q = f0 / (2*hwhm)
    f0, hwhm = 20.0, 0.5
    freqs = np.linspace(10.0, 30.0, 4000)
    amp = 1.0 / np.sqrt(1.0 + ((freqs - f0) / hwhm) ** 2)  # -3dB at f0 +- hwhm

    q, f_peak, bandwidth = quality_factor_half_power(freqs, amp)

    assert np.isclose(f_peak, f0, atol=0.05)
    assert np.isclose(bandwidth, 2.0 * hwhm, atol=0.05)
    assert np.isclose(q, f0 / (2.0 * hwhm), rtol=0.05)
