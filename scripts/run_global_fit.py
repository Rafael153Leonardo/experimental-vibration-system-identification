"""Reproduce the global Duffing fit with nonlinear damping.

Cleaned, reproducible version of the original ``prof/definitivo.ipynb`` workflow
(narrative section 11). Regenerates the ``duffing_global_envelope_fit`` figure
from the sample data, overlaying the noisy signal, the denoised signal, the
optimized numerical model and the analytic nonlinear envelope.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vibration_id.damping import hilbert_envelope
from vibration_id.global_fit import (
    fit_global_duffing,
    fit_nonlinear_envelope,
    nonlinear_envelope,
    simulate_duffing,
)
from vibration_id.pipeline import load_clean_signal
from vibration_id.spectral import dominant_frequency


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Global Duffing fit with nonlinear damping.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "data" / "sample" / "sample_vibration_18hz.csv",
        help="Input CSV with a free-vibration signal.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "figures" / "generated" / "advanced",
        help="Output directory for the generated figure.",
    )
    parser.add_argument("--fit-seconds", type=float, default=5.0, help="Window used for the global fit.")
    parser.add_argument("--max-points", type=int, default=2000, help="Decimation cap for the optimizer.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cleaned = load_clean_signal(args.csv)
    t, x_raw, x_clean, onset = cleaned.t, cleaned.raw, cleaned.clean, cleaned.onset

    # Stage 1: nonlinear envelope -> gamma, eta
    envelope = hilbert_envelope(x_clean)
    env_fit = fit_nonlinear_envelope(t, envelope)

    # Stage 2/3: global Duffing optimization (omega0_sq, beta, phi)
    f0 = dominant_frequency(t, x_clean, fmin=1.0, fmax=80.0)
    omega_guess = (2.0 * np.pi * f0) ** 2
    fit = fit_global_duffing(
        t,
        x_clean,
        gamma=env_fit.gamma,
        eta=env_fit.eta,
        omega0_sq_guess=omega_guess,
        fit_seconds=args.fit_seconds,
        max_points=args.max_points,
    )
    x_model = simulate_duffing(fit, t)
    env_model = nonlinear_envelope(t - t[0], env_fit.r0, env_fit.gamma, env_fit.eta)

    print("Global Duffing fit")
    print(f"source: {args.csv}")
    print(f"onset_index: {onset}")
    print("envelope stage:")
    print(f"  gamma: {env_fit.gamma:.5f}")
    print(f"  eta: {env_fit.eta:.5f}")
    print(f"  r0: {env_fit.r0:.4f}")
    print("global stage:")
    print(f"  omega0_sq: {fit.omega0_sq:.2f}")
    print(f"  frequency_hz: {fit.frequency_hz:.4f}")
    print(f"  beta (Duffing): {fit.beta:.4f}")
    print(f"  phi: {fit.phi:.4f} rad")
    print(f"  x0: {fit.x0:.4f}  v0: {fit.v0:.4f}")
    print(f"  mse: {fit.mse:.6f}")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t, x_raw, color="lightgray", alpha=0.6, label="Noisy signal", linewidth=0.8)
    ax.plot(t, x_clean, color="black", alpha=0.4, label="Denoised", linewidth=0.9)
    ax.plot(t, x_model, color="tab:red", linestyle="--", label="Optimized Duffing model", linewidth=1.2)
    ax.plot(t, env_model, color="tab:blue", linewidth=1.5, label="Nonlinear envelope")
    ax.plot(t, -env_model, color="tab:blue", linewidth=1.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.set_title(
        f"Global Duffing fit: f0={fit.frequency_hz:.2f} Hz, "
        f"gamma={fit.gamma:.3f}, eta={fit.eta:.3f}, beta={fit.beta:.1f}"
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out / "duffing_global_envelope_fit.png", dpi=160)
    plt.close(fig)

    print(f"figures: {args.out}")


if __name__ == "__main__":
    main()
