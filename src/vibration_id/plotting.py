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


def save_noisy_vs_filtered_plot(t, x_raw, x_clean, path: str | Path, *, detail_seconds: float = 2.0) -> None:
    """Two-panel view: full record overlay plus a zoom where the noise shows."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.asarray(t, dtype=float)

    fig, (ax_full, ax_zoom) = plt.subplots(2, 1, figsize=(12, 7))
    ax_full.plot(t, x_raw, color="gray", linewidth=0.5, alpha=0.6, label="Noisy signal")
    ax_full.plot(t, x_clean, color="tab:red", linewidth=0.7, label="Wavelet-filtered")
    ax_full.set_xlabel("Time [s]")
    ax_full.set_ylabel("Amplitude")
    ax_full.set_title("Experimental signal: noisy vs wavelet-filtered")
    ax_full.legend(loc="upper right")
    ax_full.grid(True, alpha=0.3)

    # Zoom into the decayed tail, where the quantization noise is visible
    # against the small oscillation.
    t_end = float(t[-1])
    zoom_start = min(0.45 * t_end, t_end - detail_seconds)
    mask = (t >= zoom_start) & (t <= zoom_start + detail_seconds)
    ax_zoom.plot(t[mask], np.asarray(x_raw)[mask], color="gray", linewidth=0.8, alpha=0.7, label="Noisy")
    ax_zoom.plot(t[mask], np.asarray(x_clean)[mask], color="tab:red", linewidth=1.0, label="Filtered")
    ax_zoom.set_xlabel("Time [s]")
    ax_zoom.set_ylabel("Amplitude")
    ax_zoom.set_title("Detail of the residual noise and the filtering")
    ax_zoom.legend(loc="upper right")
    ax_zoom.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


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
