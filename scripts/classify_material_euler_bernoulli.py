from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vibration_id.io import load_signal_csv
from vibration_id.materials import (
    BeamGeometry,
    rank_materials_by_frequency,
    rank_materials_by_young,
    young_modulus_from_frequency,
)
from vibration_id.preprocessing import crop_from_onset, remove_dc_offset, wavelet_denoise
from vibration_id.spectral import dominant_frequency


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify a rectangular cantilever material from an extracted natural "
            "frequency using the Euler-Bernoulli first-mode model."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "data" / "sample" / "sample_vibration_18hz.csv",
        help="Input CSV used to extract the dominant frequency by FFT.",
    )
    parser.add_argument(
        "--frequency-hz",
        type=float,
        default=None,
        help="Use a known frequency instead of extracting it from a CSV.",
    )
    parser.add_argument("--fmin", type=float, default=1.0, help="Minimum FFT frequency in Hz.")
    parser.add_argument("--fmax", type=float, default=80.0, help="Maximum FFT frequency in Hz.")
    parser.add_argument(
        "--length-m",
        type=float,
        default=0.30,
        help="Free cantilever length in meters.",
    )
    parser.add_argument(
        "--thickness-m",
        type=float,
        default=0.00233,
        help="Beam thickness in meters.",
    )
    parser.add_argument("--width-m", type=float, default=0.025, help="Beam width in meters.")
    parser.add_argument(
        "--density-kg-m3",
        type=float,
        default=1050.0,
        help="Density used for the inverse Young modulus estimate.",
    )
    parser.add_argument(
        "--tip-mass-kg",
        type=float,
        default=0.0,
        help="Mass attached at the free tip (e.g. a paper target). 0 disables the correction.",
    )
    parser.add_argument("--top-n", type=int, default=5, help="Number of material matches to print.")
    parser.add_argument("--no-crop", action="store_true", help="Do not crop from detected onset.")
    parser.add_argument("--no-denoise", action="store_true", help="Do not apply wavelet denoising.")
    return parser.parse_args()


def extract_frequency_from_csv(args: argparse.Namespace) -> float:
    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    data = load_signal_csv(args.csv, center_signal=False)
    t = data["time_s"].to_numpy()
    x = data["signal"].to_numpy()
    x = remove_dc_offset(x, mode="tail")
    if not args.no_crop:
        t, x, _ = crop_from_onset(t, x, safety_samples=5)
    if not args.no_denoise:
        x = wavelet_denoise(x, wavelet="db8", level=2)
    return dominant_frequency(t, x, fmin=args.fmin, fmax=args.fmax)


def main() -> None:
    args = parse_args()
    geometry = BeamGeometry(
        length_m=args.length_m,
        thickness_m=args.thickness_m,
        width_m=args.width_m,
    )

    if args.frequency_hz is None:
        frequency_hz = extract_frequency_from_csv(args)
        source = str(args.csv)
    else:
        frequency_hz = args.frequency_hz
        source = "manual"

    young_pa = young_modulus_from_frequency(
        geometry,
        frequency_hz=frequency_hz,
        density_kg_m3=args.density_kg_m3,
        tip_mass_kg=args.tip_mass_kg,
    )
    young_matches = rank_materials_by_young(young_pa)
    frequency_matches = rank_materials_by_frequency(geometry, frequency_hz=frequency_hz)

    print("Euler-Bernoulli material classification")
    print(f"source: {source}")
    print(f"frequency_hz: {frequency_hz:.4f}")
    print(
        "geometry: "
        f"L={geometry.length_m:.4f} m, "
        f"h={geometry.thickness_m:.5f} m, "
        f"b={geometry.width_m:.4f} m"
    )
    print(f"inverse_density_kg_m3: {args.density_kg_m3:.1f}")
    if args.tip_mass_kg > 0:
        print(f"tip_mass_kg: {args.tip_mass_kg:.6f}")
    print(f"estimated_young_gpa: {young_pa / 1e9:.3f}")
    print()
    print("Closest matches by inverse Young modulus:")
    for match in young_matches[: args.top_n]:
        status = "inside range" if match.within_range else f"{match.distance_gpa:.3f} GPa away"
        candidate = match.candidate
        print(
            f"- {candidate.name}: {candidate.young_min_gpa:.2f}-"
            f"{candidate.young_max_gpa:.2f} GPa ({status})"
        )
    print()
    print("Closest matches by predicted frequency interval:")
    for match in frequency_matches[: args.top_n]:
        status = "inside range" if match.within_range else f"{match.distance_hz:.3f} Hz away"
        candidate = match.candidate
        print(
            f"- {candidate.name}: {match.frequency_min_hz:.3f}-"
            f"{match.frequency_max_hz:.3f} Hz ({status})"
        )


if __name__ == "__main__":
    main()
