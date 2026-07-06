"""Generate the narrative figures used by the README (figures/story/).

Each figure is one act of the steel-ruler detective story:

01 - the measurement: the inox free decay and its 4.982 Hz spectral line
02 - the alibi check: measured forced-mode ratios vs the ideal cantilever
     ladder (the boundary-condition fingerprint that exonerated the clamp)
03 - the verdict: the Young-modulus ladder as the assumed thickness improves
     (documented 1.5 mm -> edge photo 1.0 mm -> micrometer 0.55 mm), against
     the material ranges; every value is recomputed here with the pipeline's
     own inverse model

The forced-mode frequencies in figure 02 are the measured values documented in
docs/ORIGINAL_CODE_AUDIT.md (the raw forced-vibration dataset is not part of
the public samples).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vibration_id.beam_modes import CANTILEVER_BETA_L
from vibration_id.materials import BeamGeometry, young_modulus_from_frequency
from vibration_id.pipeline import load_clean_signal
from vibration_id.spectral import compute_fft, dominant_frequency

STORY = ROOT / "figures" / "story"

# Measured forced-resonance frequencies (docs/ORIGINAL_CODE_AUDIT.md)
FORCED_MODES_HZ = [5.01, 31.2, 86.8, 173.5]

# Young-modulus ladder, recomputed with the pipeline's inverse model so the
# figure always matches what the code reproduces. Fixed inputs: f1 = 4.982 Hz,
# L = 0.300 m, b = 25 mm, rho = 7850 kg/m^3, 0.21 g paper tip target. Only the
# assumed thickness changes across the story.
F1_HZ = 4.982
DENSITY_KG_M3 = 7850.0
TIP_MASS_KG = 2.1e-4


def _young_modulus_gpa(thickness_mm: float) -> float:
    geometry = BeamGeometry(length_m=0.300, thickness_m=thickness_mm * 1e-3, width_m=0.025)
    modulus = young_modulus_from_frequency(
        geometry, frequency_hz=F1_HZ, density_kg_m3=DENSITY_KG_M3, tip_mass_kg=TIP_MASS_KG
    )
    return modulus / 1e9


E_WRONG_GPA = _young_modulus_gpa(1.5)  # documented thickness -> ~27 GPa
E_TRAP_GPA = _young_modulus_gpa(1.0)  # edge-photo thickness -> ~61 GPa, "looks like aluminum"
E_RIGHT_GPA = _young_modulus_gpa(0.55)  # micrometer thickness -> ~205 GPa
# Uncertainty budget on the final value (~2.6%, thickness/length dominated)
E_UNCERTAINTY_GPA = 5.3
MATERIAL_BANDS = [
    ("Acrylic / polystyrene", 2.7, 3.5),
    ("Aluminum", 68.0, 72.0),
    ("Steel (stainless/carbon)", 190.0, 210.0),
]


def fig_01_measurement() -> None:
    cleaned = load_clean_signal(ROOT / "data" / "sample" / "sample_inox_raw_calibrated.csv")
    t, x = cleaned.t, cleaned.clean
    f0 = dominant_frequency(t, x, fmin=1.0, fmax=80.0)
    freqs, amp = compute_fft(t, x)

    fig, (ax_time, ax_fft) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [3, 2]})
    mask = t <= 20.0
    ax_time.plot(t[mask], x[mask], color="gray", linewidth=0.4)
    ax_time.set_xlabel("Time [s]")
    ax_time.set_ylabel("Position [mm]")
    ax_time.set_title("The measurement: a steel ruler rings for 20+ seconds")
    ax_time.grid(True, alpha=0.3)

    band = (freqs > 0.5) & (freqs < 15.0)
    ax_fft.plot(freqs[band], amp[band], color="black", linewidth=1.0)
    ax_fft.axvline(f0, color="crimson", linestyle=":", linewidth=1)
    ax_fft.annotate(
        f"{f0:.3f} Hz\n(ensemble spread: ±0.001 Hz)",
        xy=(f0, float(np.max(amp[band]))),
        xytext=(6.5, float(np.max(amp[band])) * 0.7),
        arrowprops={"arrowstyle": "->", "color": "crimson"},
        color="crimson",
        fontsize=10,
    )
    ax_fft.set_xlabel("Frequency [Hz]")
    ax_fft.set_ylabel("FFT amplitude")
    ax_fft.set_title("One number to carry forward")
    ax_fft.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(STORY / "01_measurement.png", dpi=160)
    plt.close(fig)


def fig_02_modal_ladder() -> None:
    betas = np.asarray(CANTILEVER_BETA_L[:4])
    ideal_ratios = (betas / betas[0]) ** 2
    measured = np.asarray(FORCED_MODES_HZ)
    measured_ratios = measured / measured[0]
    modes = np.arange(1, 5)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(modes, ideal_ratios, "o-", color="black", label="ideal clamped-free ladder $\\beta_n^2/\\beta_1^2$")
    ax.plot(
        modes,
        measured_ratios,
        "s",
        color="crimson",
        markersize=9,
        fillstyle="none",
        markeredgewidth=2,
        label="measured (forced resonances)",
    )
    for n, (ideal, meas) in enumerate(zip(ideal_ratios, measured_ratios, strict=True), start=1):
        if n > 1:
            ax.annotate(f"{100 * (meas / ideal - 1):+.1f}%", xy=(n, meas), xytext=(n + 0.08, meas * 0.88), fontsize=9)
    ax.annotate(
        "a soft clamp or a tip mass\nwould push these points UP,\noff the ideal line",
        xy=(3.0, 17.5),
        xytext=(1.4, 22.0),
        arrowprops={"arrowstyle": "->", "color": "gray"},
        fontsize=10,
        color="gray",
    )
    ax.set_yscale("log")
    ax.set_xticks(modes)
    ax.set_xlabel("Mode number $n$")
    ax.set_ylabel("$f_n / f_1$  (log)")
    ax.set_title(
        "The alibi check: mode ratios are a boundary-condition fingerprint —\nand the clamp walks free (deviations ~1%)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(STORY / "02_modal_ladder.png", dpi=160)
    plt.close(fig)


def fig_03_verdict() -> None:
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for label, lo, hi in MATERIAL_BANDS:
        ax.axvspan(lo, hi, alpha=0.18, color="tab:blue")
        ax.text(np.sqrt(lo * hi), 1.62, label, ha="center", fontsize=9, color="tab:blue")

    ax.errorbar(
        [E_RIGHT_GPA],
        [1.0],
        xerr=[[E_UNCERTAINTY_GPA], [E_UNCERTAINTY_GPA]],
        fmt="o",
        color="tab:green",
        markersize=9,
        capsize=5,
        label=f"micrometer thickness 0.55 mm: {E_RIGHT_GPA:.1f} ± {E_UNCERTAINTY_GPA:.1f} GPa",
    )
    ax.plot(
        [E_WRONG_GPA],
        [1.0],
        "X",
        color="crimson",
        markersize=12,
        label=f"documented thickness 1.5 mm: {E_WRONG_GPA:.0f} GPa — no material lives here",
    )
    ax.plot(
        [E_TRAP_GPA],
        [1.0],
        "X",
        color="darkorange",
        markersize=12,
        label=f"edge-photo thickness 1.0 mm: {E_TRAP_GPA:.0f} GPa — looks like aluminum",
    )
    ax.annotate(
        "E ∝ 1/h²: the missing factor of 7.6\nwas hiding in the thickness",
        xy=(E_WRONG_GPA, 1.0),
        xytext=(4.0, 1.30),
        arrowprops={"arrowstyle": "->", "color": "crimson"},
        fontsize=10,
        color="crimson",
    )
    ax.annotate(
        "the trap: just below aluminum —\nplausible, and wrong",
        xy=(E_TRAP_GPA, 0.97),
        xytext=(28.0, 0.70),
        arrowprops={"arrowstyle": "->", "color": "darkorange"},
        fontsize=10,
        color="darkorange",
    )
    ax.set_xscale("log")
    ax.set_xlim(2, 300)
    ax.set_ylim(0.6, 1.8)
    ax.set_yticks([])
    ax.set_xlabel("Young modulus [GPa] (log)")
    ax.set_title("The verdict: same frequency, same physics — the geometry was lying")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3, axis="x", which="both")
    fig.tight_layout()
    fig.savefig(STORY / "03_verdict.png", dpi=160)
    plt.close(fig)


def main() -> None:
    STORY.mkdir(parents=True, exist_ok=True)
    fig_01_measurement()
    fig_02_modal_ladder()
    fig_03_verdict()
    for name in ["01_measurement", "02_modal_ladder", "03_verdict"]:
        print(f"figure: {STORY / (name + '.png')}")


if __name__ == "__main__":
    main()
