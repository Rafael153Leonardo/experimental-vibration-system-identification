from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vibration_id.havok import havok_svd, identify_havok
from vibration_id.pinn import PinnTrainingConfig, pinn_available, predict, train_two_stage_pinn
from vibration_id.pipeline import decimate, load_clean_signal
from vibration_id.sindy_model import fit_linear_sindy, physical_params_from_sindy, state_from_position


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run advanced SINDy, HAVOK and PINN analysis.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "data" / "sample" / "sample_vibration_18hz.csv",
        help="Input CSV with a vibration signal.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "figures" / "generated" / "advanced",
        help="Output directory for generated advanced figures.",
    )
    parser.add_argument("--max-samples", type=int, default=3000, help="Maximum samples for advanced methods.")
    parser.add_argument(
        "--sindy-samples",
        type=int,
        default=9000,
        help="Contiguous full-rate samples for SINDy (derivative-based identification needs the full sampling rate).",
    )
    parser.add_argument("--havok-delays", type=int, default=120, help="Number of Hankel delays.")
    parser.add_argument("--havok-rank", type=int, default=6, help="HAVOK reduced rank.")
    parser.add_argument("--run-pinn", action="store_true", help="Train the optional PyTorch PINN model.")
    parser.add_argument(
        "--pinn-csv",
        type=Path,
        default=ROOT / "data" / "sample" / "sample_inox_synchronized.csv",
        help="Signal for the PINN demo (defaults to the inox 4.98 Hz sample used by the original experiment).",
    )
    parser.add_argument("--pinn-stage1-epochs", type=int, default=3000, help="PINN linear-stage epochs.")
    parser.add_argument("--pinn-stage2-epochs", type=int, default=1000, help="PINN nonlinear-stage epochs.")
    parser.add_argument("--pinn-max-points", type=int, default=1200, help="Maximum PINN training samples.")
    return parser.parse_args()


def save_havok_figures(out: Path, t: np.ndarray, singular_values: np.ndarray, forcing: np.ndarray) -> None:
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(np.arange(1, min(len(singular_values), 30) + 1), singular_values[:30], marker="o")
    ax.set_xlabel("Mode")
    ax.set_ylabel("Singular value")
    ax.set_title("HAVOK singular value spectrum")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "havok_singular_values.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t[: len(forcing)], forcing, color="black", linewidth=1)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("u(t)")
    ax.set_title("HAVOK internal forcing")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "havok_forcing.png", dpi=160)
    plt.close(fig)


def save_sindy_plot(out: Path, t: np.ndarray, x: np.ndarray, simulation: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, x, color="black", alpha=0.35, label="Real signal")
    ax.plot(t[: len(simulation)], simulation, color="tab:red", linestyle="--", label="SINDy simulation")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Signal")
    ax.set_title("SINDy linear oscillator simulation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "sindy_simulation.png", dpi=160)
    plt.close(fig)


def save_pinn_plot(out: Path, t: np.ndarray, x: np.ndarray, x_pred: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, x, color="black", alpha=0.25, label="Real signal")
    ax.plot(t, x_pred, color="tab:red", linestyle="--", label="PINN")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Signal")
    ax.set_title("Two-stage PINN fit")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "pinn_fit.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cleaned = load_clean_signal(args.csv)
    t, x, onset = cleaned.t, cleaned.clean, cleaned.onset
    t_small, x_small = decimate(t, x, args.max_samples)
    print("Advanced vibration analysis")
    print(f"source: {args.csv}")
    print(f"onset_index: {onset}")
    print(f"samples_used: {len(t_small)}")

    delays = min(args.havok_delays, max(10, len(x_small) // 4))
    H, U, S, Vt = havok_svd(x_small, delays=delays)
    rank = min(args.havok_rank, len(S), delays)
    A, B, z_state, u_state = identify_havok(t_small, S, Vt, rank=rank)
    energy = float(np.sum(S[:rank] ** 2) / np.sum(S**2))
    save_havok_figures(args.out, t_small, S, u_state.ravel())
    print("HAVOK")
    print(f"  hankel_shape: {H.shape}")
    print(f"  rank: {rank}")
    print(f"  retained_energy: {energy:.4f}")
    print(f"  A_shape: {A.shape}")
    print(f"  B_shape: {B.shape}")

    try:
        # SINDy works on a contiguous full-rate window: strided decimation
        # leaves too few samples per period, and finite-difference derivatives
        # on such a grid bias the identified stiffness strongly low. The savgol
        # window must also stay well under one oscillation period.
        t_id = t[: args.sindy_samples]
        x_id = x[: args.sindy_samples]
        state = state_from_position(t_id, x_id, method="savgol", window_length=21, polyorder=3)
        dt = float(np.median(np.diff(t_id)))
        model = fit_linear_sindy([state], dt=dt, threshold=0.01, alpha=0.05)
        params = physical_params_from_sindy(model)
        sim = model.simulate(state[0], t_id)
        save_sindy_plot(args.out, t_id, x_id, sim[:, 0])
        print("SINDy")
        print(f"  frequency_hz: {params.frequency_hz:.4f}")
        print(f"  zeta: {params.zeta:.6f}")
        print(f"  stiffness_over_mass: {params.stiffness_over_mass:.4f}")
        print(f"  damping_over_mass: {params.damping_over_mass:.6f}")
    except ImportError as exc:
        print(f"SINDy skipped: {exc}")
    except Exception as exc:
        print(f"SINDy failed: {exc}")

    if args.run_pinn:
        if not pinn_available():
            print("PINN skipped: install torch or requirements-advanced.txt.")
        else:
            config = PinnTrainingConfig(
                stage1_epochs=args.pinn_stage1_epochs,
                stage2_epochs=args.pinn_stage2_epochs,
                max_points=args.pinn_max_points,
            )
            # The PINN gets the full-rate signal; train_two_stage_pinn picks a
            # Nyquist-safe training window itself.
            pinn_cleaned = load_clean_signal(args.pinn_csv)
            result = train_two_stage_pinn(pinn_cleaned.t, pinn_cleaned.clean, config)
            x_pred = predict(result.model, result.t_train)
            save_pinn_plot(args.out, result.t_train, result.x_train, x_pred)
            print("PINN")
            print(f"  source: {args.pinn_csv}")
            print(f"  device: {result.device}")
            print(f"  training_window_s: {float(result.t_train[-1] - result.t_train[0]):.2f}")
            print(f"  frequency_hz: {result.frequency_hz:.4f}")
            print(f"  mu: {result.mu:.6f}")
            print(f"  alpha: {result.alpha:.8f}")
            print(f"  stiffness_over_mass: {result.stiffness_over_mass:.4f}")
    else:
        print("PINN skipped: pass --run-pinn after installing requirements-advanced.txt.")

    print(f"figures: {args.out}")


if __name__ == "__main__":
    main()
