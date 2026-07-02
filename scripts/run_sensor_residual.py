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

from vibration_id.pipeline import load_clean_signal
from vibration_id.preprocessing import velocity_savgol, wavelet_denoise
from vibration_id.sensor_residual import (
    evaluate_residual_surface,
    fit_residual_surface,
    fit_static_nonlinearity,
    linear_dynamics_residual,
    simulate_linear_state,
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
    parser.add_argument(
        "--max-samples",
        type=int,
        default=40000,
        help="Contiguous full-rate samples to analyze (derivative-based fits need a uniform grid).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cleaned = load_clean_signal(args.csv, denoise=False)
    # A contiguous full-rate window, NOT a strided decimation: the previous
    # linspace-based decimation produced a non-uniform grid (stride jitter)
    # that biased the identified stiffness ~35% low (4.04 Hz vs the 4.98 Hz
    # spectral peak).
    t = cleaned.t[: args.max_samples]
    x = cleaned.raw[: args.max_samples]
    onset = cleaned.onset
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

    # 4. Static nonlinearity: position residual vs simulated linear state
    #    (the original testes.py analysis behind sensor_nonlinearity_fit.png)
    q_sim, _ = simulate_linear_state(res.omega0_sq, res.gamma, q0=float(q[0]), v0=float(v[0]), t=t)
    position_residual = x - q_sim
    h_coeffs = fit_static_nonlinearity(q_sim, position_residual, degree=3)
    h0, h1, h2, h3 = h_coeffs
    print("static nonlinearity fit h(x) to the position residual:")
    print(f"  a0={h0:.4e} a1={h1:.4e} a2={h2:.4e} a3={h3:.4e}")

    x_grid = np.linspace(float(np.min(q_sim)), float(np.max(q_sim)), 400)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(q_sim, position_residual, s=2, alpha=0.2, color="tab:blue", label="Position residual")
    ax.plot(x_grid, np.polynomial.polynomial.polyval(x_grid, h_coeffs), "r", linewidth=2, label="h(x) cubic fit")
    ax.set_xlabel("Simulated linear state $x_{model}$")
    ax.set_ylabel("Residual $r$")
    ax.set_title("Static sensor-nonlinearity fit")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out / "sensor_nonlinearity_fit.png", dpi=160)
    plt.close(fig)

    # 5. Response-surface slice r(q, v=0)
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
