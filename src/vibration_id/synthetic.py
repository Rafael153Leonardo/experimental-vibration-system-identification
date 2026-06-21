from __future__ import annotations

import numpy as np


def damped_oscillator(
    *,
    duration: float = 30.0,
    fs: float = 1000.0,
    frequency_hz: float = 18.7,
    gamma: float = 0.42,
    amplitude: float = 4.5,
    noise_std: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a damped oscillator similar to the experimental signal."""

    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration, 1.0 / fs)
    envelope = amplitude * np.exp(-gamma * t / 2.0)
    x = envelope * np.cos(2.0 * np.pi * frequency_hz * t)
    x += rng.normal(0.0, noise_std, size=len(t))
    return t, x
