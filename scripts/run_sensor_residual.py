"""Reproduce the linear-model residual (sensor nonlinearity) analysis.

Cleaned, reproducible version of the residual analysis in the original
``completo.py`` (narrative section 4), with the ``DT=1000`` savgol bug fixed.
Regenerates the ``figures/sensor`` residual story: time signature, spectrum,
phase-space portrait and the response-surface slice ``r(q, v)``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vibration_id.io import load_signal_csv
from vibration_id.preprocessing import crop_from_onset, remove_dc_offset, velocity_savgol, wavelet_denoise
from vibration_id.sensor_residual import (
    evaluate_residual_surface,
    fit_residual_surface,
    linear_dynamics_residual,
)
from vibration_id.spectral import compute_fft


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Linear-model residual / sensor nonlinearity analysis.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "data" / "sample" / "sample_inox_synchronized.csv",
        help="Input CSV with a free-vibration signal.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "figures" / "generated" / "sensor",
        help="Output directory for generated figures.",
    )
    parser.add_argument("--max-samples", type=int, default=20000, help="Limit samples for speed.")
    return parser.parse_args()


def downsample(t: np.ndarray, x: np.ndarray, max_samples: int) -> tuple[np.ndarray, np.ndarray]:
    if max_samples <= 0 or len(t) <= max_samples:
        return t, x
    idx = np.linspace(0, len(t) - 1, max_samples).astype(int)
    return t[idx], x[idx]


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    data = load_signal_csv(args.csv, center_signal=False)
    t = data["time_s"].to_numpy()
    x = remove_dc_offset(data["signal"].to_numpy(), mode="tail")
    t, x, onset = crop_from_onset(t, x, safety_samples=5)
    t, x = downsample(t, x, args.max_samples)
    dt = float(np.median(np.diff(t)))

    q = wavelet_denoise(x, wavelet="db8", level=2)
    v = velocity_savgol(t, q, window_length=81, polyorder=3)  # correct dt via t (no DT=1000 bug)
    res = linear_dynamics_residual(q, v, dt=dt)
    surface = fit_residual_surface(res.q, res.v, res.residual)

    print("Sensor residual analysis")
    print(f"source: {args.csv}")
    print(f"onset_index: {onset}")
    print("linear model:")
    print(f"  omega0_sq: {res.omega0_sq:.4f}")
    print(f"  frequency_hz: {res.frequency_hz:.4f}")
    print(f"  gamma: {res.gamma:.6f}")
    print(f"  residual_energy_fraction: {res.residual_energy_fraction:.4f}")
    c_q2, c_q3, c_qv, c_v2 = surface
    print("residual response surface r(q,v):")
    print(f"  q^2={c_q2:.4e} q^3={c_q3:.4e} q*v={c_qv:.4e} v^2={c_v2:.4e}")

    # 1. Residual time signature
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, res.residual, color="crimson", linewidth=0.8)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Residual acceleration")
    ax.set_title("Linear-model residual signature")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out / "sensor_residual_signature.png", dpi=160)
    plt.close(fig)

    # 2. Residual spectrum
    freqs, amp = compute_fft(t, res.residual)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogy(freqs, amp + 1e-12, color="tab:purple", linewidth=0.9)
    ax.set_xlim(0, 60)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Amplitude")
    ax.set_title("Residual spectrum")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out / "sensor_residual_spectrum.png", dpi=160)
    plt.close(fig)

    # 3. Residual phase space (residual vs displacement, colored by velocity)
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(res.q, res.residual, c=res.v, cmap="plasma", s=2, alpha=0.5)
    ax.set_xlabel("Displacement q")
    ax.set_ylabel("Residual")
    ax.set_title("Residual phase space")
    fig.colorbar(sc, label="velocity v")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out / "sensor_residual_phase_space.png", dpi=160)
    plt.close(fig)

    # 4. Response-surface slice r(q, v=0)
    q_grid = np.linspace(float(np.min(res.q)), float(np.max(res.q)), 400)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(q_grid, evaluate_residual_surface(surface, q_grid, np.zeros_like(q_grid)), color="tab:blue")
    ax.set_xlabel("Displacement q")
    ax.set_ylabel("Residual r(q, v=0)")
    ax.set_title("Residual response surface slice")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out / "sensor_response_surface_slice.png", dpi=160)
    plt.close(fig)

    print(f"figures: {args.out}")


if __name__ == "__main__":
    main()
