"""Backbone curve and damping-law diagnostics for the inox free decay.

Answers two questions the modulus story leaves implicit -- *where* the system's
nonlinearity lives:

* the **backbone** (instantaneous frequency vs amplitude) is flat, so the beam
  stiffness is linear and the cubic term in the SINDy / global-Duffing fits is a
  sensor-map artifact, not beam physics;
* the ring-down **envelope** is fit noticeably better by amplitude-dependent
  damping than by a single exponential -- the nonlinearity is in the dissipation.

Regenerates ``figures/generated/advanced/backbone_damping.png`` from the shipped
inox sample, so it runs from a clean checkout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vibration_id.backbone import analyze_ringdown
from vibration_id.pipeline import load_clean_signal
from vibration_id.spectral import dominant_frequency


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backbone curve and damping-law diagnostics.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "data" / "sample" / "sample_inox_raw_calibrated.csv",
        help="Input CSV with a free-decay (ring-down) signal.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "figures" / "generated" / "advanced",
        help="Output directory for the generated figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cleaned = load_clean_signal(args.csv)
    t, x = cleaned.t, cleaned.clean
    f0 = dominant_frequency(t, x, fmin=1.0, fmax=80.0)
    analysis = analyze_ringdown(t, x, frequency_hz=f0)
    bb, dmp = analysis.backbone, analysis.damping

    print("Backbone + damping diagnostics")
    print(f"source: {args.csv}")
    print(f"dominant frequency: {f0:.4f} Hz")
    print("backbone (frequency vs amplitude):")
    print(f"  f0 (extrapolated to A->0): {bb.f0_hz:.4f} Hz")
    print(f"  slope df/d(A^2):           {bb.slope_hz_per_amp2:+.3e} Hz/amp^2")
    print(f"  fractional pull over swing: {bb.fractional_pull * 100:+.3f} %  (<1% => linear stiffness)")
    print(f"  corr(A^2, f):              {bb.pearson_r:+.3f}")
    print("damping law (envelope fits):")
    print(f"  linear-viscous:  gamma={dmp.gamma_linear:.4f} 1/s  Q={dmp.q_linear:.0f}  R^2={dmp.r2_linear:.4f}")
    print(
        f"  amplitude-dep.:  gamma={dmp.gamma_nonlinear:.4f} 1/s  "
        f"eta={dmp.eta_nonlinear:.4f}  R^2={dmp.r2_nonlinear:.4f}"
    )
    print(f"  data prefer {'AMPLITUDE-DEPENDENT' if dmp.prefers_nonlinear else 'LINEAR'} damping")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # Panel 1: envelope with both damping fits.
    ax = axes[0]
    ax.plot(analysis.t_window, analysis.envelope, color="0.5", linewidth=1.2, label="peak envelope")
    ax.plot(
        analysis.t_window,
        analysis.linear_model,
        color="tab:orange",
        linestyle="--",
        linewidth=1.6,
        label=f"linear viscous ($R^2$={dmp.r2_linear:.3f})",
    )
    ax.plot(
        analysis.t_window,
        analysis.nonlinear_model,
        color="tab:blue",
        linewidth=1.6,
        label=f"amplitude-dependent ($R^2$={dmp.r2_nonlinear:.3f})",
    )
    ax.set_xlabel("Time in decay [s]")
    ax.set_ylabel("Amplitude [mm]")
    ax.set_title(f"Damping law  (Q$\\approx${dmp.q_linear:.0f})")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: backbone -- instantaneous frequency vs amplitude (binned).
    ax = axes[1]
    ax.errorbar(
        analysis.amplitude_bins,
        analysis.frequency_bins,
        yerr=analysis.frequency_bin_err,
        fmt="o",
        color="0.35",
        markersize=4,
        elinewidth=1.0,
        capsize=2,
        label="binned mean $\\pm$ 1$\\sigma$",
    )
    a_grid = np.linspace(bb.amplitude_lo, bb.amplitude_hi, 50)
    ax.plot(
        a_grid,
        bb.f0_hz + bb.slope_hz_per_amp2 * a_grid**2,
        color="tab:red",
        linewidth=2.0,
        label=f"fit: pull={bb.fractional_pull * 100:+.2f}%",
    )
    ax.axhline(bb.f0_hz, color="k", linewidth=0.8, linestyle=":")
    ax.set_ylim(bb.f0_hz - 0.12, bb.f0_hz + 0.12)
    ax.set_xlabel("Amplitude [mm]")
    ax.set_ylabel("Instantaneous frequency [Hz]")
    ax.set_title(
        f"Backbone nearly flat  ($|\\Delta f/f|${abs(bb.fractional_pull) * 100:.1f}% "
        f"over 3$\\times$ decay $\\Rightarrow$ linear stiffness)"
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: envelope-fit residuals -- the linear model leaves structure.
    ax = axes[2]
    res_lin = analysis.envelope - analysis.linear_model
    res_nl = analysis.envelope - analysis.nonlinear_model
    ax.plot(analysis.t_window, res_lin, color="tab:orange", linewidth=1.0, label="linear-viscous residual")
    ax.plot(analysis.t_window, res_nl, color="tab:blue", linewidth=1.0, label="amplitude-dep. residual")
    ax.axhline(0.0, color="k", linewidth=0.8)
    ax.set_xlabel("Time in decay [s]")
    ax.set_ylabel("Envelope residual [mm]")
    ax.set_title("Residuals: linear leaves curvature")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Where the nonlinearity lives: linear stiffness, nonlinear damping", fontsize=13, y=1.02)
    fig.tight_layout()
    out_path = args.out / "backbone_damping.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"figures: {args.out}")


if __name__ == "__main__":
    main()
