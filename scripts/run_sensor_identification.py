"""Reproduce the dynamics + sensor-output-map identification.

This is the cleaned, reproducible version of the original ``sss.ipynb`` workflow
(see ``docs/ORIGINAL_CODE_AUDIT.md``). It separates the mechanical Duffing
dynamics from the static sensor nonlinearity ``h(q)`` and saves the figures that
back the ``figures/sensor`` story.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vibration_id.nonlinear_id import identify_duffing, identify_sensor_map, simulate_model
from vibration_id.pipeline import decimate, load_clean_signal
from vibration_id.preprocessing import velocity_savgol, wavelet_denoise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Duffing dynamics + sensor output-map identification.")
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


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cleaned = load_clean_signal(args.csv, denoise=False)
    t, x_raw = decimate(cleaned.t, cleaned.raw, args.max_samples)
    onset = cleaned.onset
    dt = float(np.median(np.diff(t)))

    # The denoised displacement is treated as the mechanical state q; the raw
    # signal is the sensor measurement we want the output map h(q) to reproduce.
    q = wavelet_denoise(x_raw, wavelet="db8", level=2)
    v = velocity_savgol(t, q, window_length=81, polyorder=3)

    params = identify_duffing(q, v, dt=dt)
    sensor = identify_sensor_map(q, x_raw)
    q_sim, y_sim = simulate_model(params, sensor, q0=float(q[0]), v0=float(v[0]), t=t)

    print("Sensor + dynamics identification")
    print(f"source: {args.csv}")
    print(f"onset_index: {onset}")
    print(f"samples_used: {len(t)}")
    print("Duffing dynamics:")
    print(f"  omega0_sq: {params.omega0_sq:.4f}")
    print(f"  frequency_hz: {params.frequency_hz:.4f}")
    print(f"  gamma: {params.gamma:.6f}")
    print(f"  alpha (cubic): {params.alpha:.6f}")
    print("Sensor output map h(q) = a0 + a1 q + a2 q^2 + a3 q^3:")
    a0, a1, a2, a3 = sensor.coefficients
    print(f"  a0={a0:.4e} a1={a1:.4e} a2={a2:.4e} a3={a3:.4e}")
    print(f"  nonlinearity_strength: {sensor.nonlinearity_strength:.4f}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, x_raw, color="black", alpha=0.3, label="Measured signal")
    ax.plot(t, y_sim, color="tab:red", linestyle="--", label="Model (dynamics + sensor)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Sensor reading")
    ax.set_title("Duffing dynamics with sensor output map")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out / "dynamics_plus_sensor_fit.png", dpi=160)
    plt.close(fig)

    q_grid = np.linspace(float(np.min(q)), float(np.max(q)), 400)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(q_grid, sensor(q_grid), color="tab:blue", label="h(q) identified")
    ax.plot(q_grid, q_grid, color="gray", linestyle=":", label="linear reference")
    ax.set_xlabel("True displacement q")
    ax.set_ylabel("Sensor output h(q)")
    ax.set_title("Identified sensor nonlinearity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out / "sensor_output_map.png", dpi=160)
    plt.close(fig)

    print(f"figures: {args.out}")


if __name__ == "__main__":
    main()
