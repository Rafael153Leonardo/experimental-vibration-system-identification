from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_signal_plot(t, x, path: str | Path, *, title: str = "Signal") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(t, x, linewidth=0.7)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_fft_plot(freqs, amp, path: str | Path, *, fmax: float = 80.0) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(freqs, amp, linewidth=0.8)
    plt.xlim(0, fmax)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.title("FFT spectrum")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_envelope_plot(t, x, envelope, model, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(t, x, alpha=0.25, label="signal")
    plt.plot(t, envelope, label="Hilbert envelope")
    plt.plot(t, model, "--", label="exponential fit")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Envelope fit")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_scalogram_plot(t, freqs, power, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    extent = [float(np.min(t)), float(np.max(t)), float(np.min(freqs)), float(np.max(freqs))]
    plt.imshow(power, extent=extent, aspect="auto", origin="lower", interpolation="nearest")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.title("CWT scalogram")
    plt.colorbar(label="|coef|")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

