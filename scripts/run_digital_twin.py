"""Digital twin of the ruler rig: validation + virtual experiments.

Seeds the twin with the identified parameters (E = 205.3 GPa, L = 300 mm,
h = 0.55 mm, 0.21 g tip target, calibrated nonlinear damping and cubic sensor
map), then

1. **validates** it against the real inox sample -- the synthetic ring-down,
   run through the same analysis pipeline, must recover the measured frequency,
   quality factor, flat backbone and amplitude-dependent damping; and

2. runs the four **virtual experiments**: a free-decay pluck, a forced resonance
   sweep (the beta_n^2 modal ladder), a geometry/material what-if (the 0.55 mm
   vs 1.5 mm trap), and tip-mass sensing (frequency shift vs added mass).

Regenerates ``figures/generated/advanced/digital_twin.png`` from the shipped
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

from vibration_id.backbone import compare_damping_models
from vibration_id.digital_twin import DigitalTwin
from vibration_id.materials import BeamGeometry, young_modulus_from_frequency
from vibration_id.pipeline import load_clean_signal
from vibration_id.spectral import compute_fft, dominant_frequency


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Digital twin validation and virtual experiments.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "data" / "sample" / "sample_inox_raw_calibrated.csv",
        help="Real free-decay record to validate the twin against.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "figures" / "generated" / "advanced",
        help="Output directory for the generated figure.",
    )
    return parser.parse_args()


def _signature(t: np.ndarray, x: np.ndarray) -> dict[str, float]:
    f0 = dominant_frequency(t, x, fmin=1.0, fmax=80.0)
    dmp, *_ = compare_damping_models(t, x, frequency_hz=f0)
    return {"f0": f0, "Q": dmp.q_linear, "r2_lin": dmp.r2_linear, "r2_nl": dmp.r2_nonlinear}


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # -- real reference ------------------------------------------------------
    real = load_clean_signal(args.csv)
    tr, xr = real.t, real.clean
    sig_real = _signature(tr, xr)

    # -- the twin, seeded with the identified parameters ---------------------
    # Replay this record: match the twin's initial state to the real onset so
    # the two track cycle-by-cycle (the onset crop begins near a zero crossing,
    # so most of the initial energy is in the velocity).
    fs_real = 1.0 / float(np.median(np.diff(tr)))
    q0 = float(xr[0])
    v0 = float((xr[1] - xr[0]) * fs_real)
    twin = DigitalTwin()
    pluck = twin.simulate_free_decay(amplitude_mm=q0, v0=v0, duration_s=float(tr[-1]), fs=fs_real)
    tt, xt = pluck.t, pluck.signal
    sig_twin = _signature(tt, xt)

    print("Digital twin -- validation against the real inox sample")
    print(f"{'quantity':16s} {'real':>12s} {'twin':>12s}")
    print(f"{'f0 [Hz]':16s} {sig_real['f0']:12.4f} {sig_twin['f0']:12.4f}")
    print(f"{'Q':16s} {sig_real['Q']:12.0f} {sig_twin['Q']:12.0f}")
    print(f"{'R^2 linear':16s} {sig_real['r2_lin']:12.3f} {sig_twin['r2_lin']:12.3f}")
    print(f"{'R^2 nonlinear':16s} {sig_real['r2_nl']:12.3f} {sig_twin['r2_nl']:12.3f}")
    print(f"modal ladder [Hz]: {np.round(twin.natural_frequencies(4), 1)}")
    print(f"Young estimate: {twin.young_modulus_estimate() / 1e9:.1f} GPa -> {twin.material_verdict()}")

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))

    # (1) ring-down overlay (first seconds)
    ax = axes[0, 0]
    win = tr < 4.0
    ax.plot(tr[win], xr[win], color="0.55", linewidth=0.8, label="real ruler")
    winb = tt < 4.0
    ax.plot(tt[winb], xt[winb], color="tab:red", linewidth=0.8, alpha=0.8, label="digital twin")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Sensor signal [mm]")
    ax.set_title("Pluck / free decay: real vs twin")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # (2) spectra
    ax = axes[0, 1]
    fr, ar = compute_fft(tr, xr)
    ft, at = compute_fft(tt, xt)
    ax.semilogy(fr, ar, color="0.55", linewidth=0.8, label="real")
    ax.semilogy(ft, at, color="tab:red", linewidth=0.8, alpha=0.8, label="twin")
    ax.set_xlim(0, 40)
    ax.set_ylim(max(ar.max(), at.max()) * 1e-4, max(ar.max(), at.max()) * 2)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Spectrum: real {sig_real['f0']:.3f} Hz vs twin {sig_twin['f0']:.3f} Hz")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # (3) damping validation: Q and R^2 side by side
    ax = axes[0, 2]
    labels = ["real", "twin"]
    q_vals = [sig_real["Q"], sig_twin["Q"]]
    ax.bar([0, 1], q_vals, color=["0.55", "tab:red"], width=0.6)
    for i, q in enumerate(q_vals):
        ax.text(i, q, f"Q={q:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_ylabel("Quality factor Q")
    ax.set_ylim(0, max(q_vals) * 1.25)
    ax.set_title("Damping: quality factor match")
    ax.grid(True, axis="y", alpha=0.3)

    # (4) forced resonance sweep: the modal ladder
    ax = axes[1, 0]
    fsweep = np.linspace(2.0, 200.0, 6000)
    mag, _ = twin.frequency_response(fsweep, n_modes=4)
    ax.semilogy(fsweep, mag / mag.max(), color="tab:blue", linewidth=1.0)
    modes = twin.natural_frequencies(4)
    for m in modes:
        ax.axvline(m, color="k", linestyle=":", linewidth=0.7)
    ax.set_xlabel("Drive frequency [Hz]")
    ax.set_ylabel("Response (normalized)")
    ax.set_title(f"Forced sweep: modal ladder {np.round(modes, 0).astype(int)}")
    ax.grid(True, alpha=0.3)

    # (5) geometry what-if: apparent modulus vs assumed thickness
    ax = axes[1, 1]
    thickness_mm = np.linspace(0.4, 1.6, 200)
    f1 = twin.fundamental_hz()  # the measured frequency is fixed
    e_apparent = np.array(
        [
            young_modulus_from_frequency(
                BeamGeometry(length_m=0.30, thickness_m=h * 1e-3, width_m=0.025),
                frequency_hz=f1,
                density_kg_m3=7850.0,
                tip_mass_kg=0.21e-3,
            )
            / 1e9
            for h in thickness_mm
        ]
    )
    ax.plot(thickness_mm, e_apparent, color="tab:green", linewidth=1.5)
    ax.axhspan(190, 210, color="tab:gray", alpha=0.25, label="steel band")
    ax.axvline(0.55, color="tab:red", linestyle="--", linewidth=1.0, label="0.55 mm (micrometer)")
    ax.axvline(1.5, color="k", linestyle=":", linewidth=1.0, label="1.5 mm (assumed)")
    ax.set_xlabel("Assumed thickness [mm]")
    ax.set_ylabel("Apparent Young modulus [GPa]")
    ax.set_title("What-if geometry: E $\\propto$ 1/h$^2$")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # (6) tip-mass sensing: frequency shift vs added mass
    ax = axes[1, 2]
    added_g = np.linspace(0.0, 2.0, 60)
    f_shift = twin.tip_mass_frequency_shift(added_g * 1e-3)  # grams -> kg
    ax.plot(added_g, f_shift, color="tab:purple", linewidth=1.5)
    # local sensitivity near zero added mass; Hz/g is numerically mHz/mg.
    sens = (f_shift[0] - f_shift[1]) / (added_g[1] - added_g[0])
    ax.set_xlabel("Added tip mass [g]")
    ax.set_ylabel("Fundamental frequency [Hz]")
    ax.set_title(f"Mass sensing: ~{sens:.2f} mHz/mg near baseline")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Digital twin of the ruler rig: validated forward model + virtual experiments",
        fontsize=14,
        y=1.00,
    )
    fig.tight_layout()
    out_path = args.out / "digital_twin.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"figures: {args.out}")


if __name__ == "__main__":
    main()
