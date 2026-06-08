from __future__ import annotations

import argparse
import csv
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
        description="Run a metadata-driven Euler-Bernoulli material study."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=ROOT / "data" / "sample" / "material_trials.csv",
        help="CSV with trial metadata.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "results" / "material_classification_summary.csv",
        help="Output summary CSV.",
    )
    parser.add_argument("--fmin", type=float, default=1.0, help="Minimum FFT frequency in Hz.")
    parser.add_argument("--fmax", type=float, default=80.0, help="Maximum FFT frequency in Hz.")
    parser.add_argument("--top-n", type=int, default=3, help="Number of matches to store.")
    parser.add_argument("--no-crop", action="store_true", help="Do not crop from detected onset.")
    parser.add_argument("--no-denoise", action="store_true", help="Do not apply wavelet denoising.")
    return parser.parse_args()


def parse_optional_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    return float(value)


def extract_frequency(path: Path, args: argparse.Namespace) -> float:
    data = load_signal_csv(path, center_signal=False)
    t = data["time_s"].to_numpy()
    x = data["signal"].to_numpy()
    x = remove_dc_offset(x, mode="tail")
    if not args.no_crop:
        t, x, _ = crop_from_onset(t, x, safety_samples=5)
    if not args.no_denoise:
        x = wavelet_denoise(x, wavelet="db8", level=2)
    return dominant_frequency(t, x, fmin=args.fmin, fmax=args.fmax)


def format_matches(matches: list[object], attr: str, top_n: int) -> str:
    names = [getattr(match, attr).name for match in matches[:top_n]]
    return "; ".join(names)


def analyze_row(row: dict[str, str], args: argparse.Namespace) -> dict[str, str]:
    file_path = row.get("file_path", "").strip()
    frequency_hz = parse_optional_float(row.get("frequency_hz"))
    source_file = ""

    if frequency_hz is None and file_path:
        source_file = file_path
        frequency_hz = extract_frequency(ROOT / file_path, args)
    elif frequency_hz is None:
        return {
            **row,
            "computed_frequency_hz": "",
            "estimated_young_gpa": "",
            "closest_young_matches": "",
            "closest_frequency_matches": "",
            "analysis_status": "missing_frequency",
        }

    length_m = parse_optional_float(row.get("length_m"))
    thickness_m = parse_optional_float(row.get("thickness_m"))
    width_m = parse_optional_float(row.get("width_m"))
    density_kg_m3 = parse_optional_float(row.get("density_kg_m3"))

    result = {
        **row,
        "source_file": source_file,
        "computed_frequency_hz": f"{frequency_hz:.6f}",
        "estimated_young_gpa": "",
        "closest_young_matches": "",
        "closest_frequency_matches": "",
        "analysis_status": "frequency_only",
    }

    if length_m is None or thickness_m is None or density_kg_m3 is None:
        missing = []
        if length_m is None:
            missing.append("length_m")
        if thickness_m is None:
            missing.append("thickness_m")
        if density_kg_m3 is None:
            missing.append("density_kg_m3")
        result["analysis_status"] = "missing_" + "_".join(missing)
        return result

    geometry = BeamGeometry(
        length_m=length_m,
        thickness_m=thickness_m,
        width_m=width_m if width_m is not None else 0.025,
    )
    young_pa = young_modulus_from_frequency(
        geometry,
        frequency_hz=frequency_hz,
        density_kg_m3=density_kg_m3,
    )
    young_matches = rank_materials_by_young(young_pa)
    frequency_matches = rank_materials_by_frequency(geometry, frequency_hz=frequency_hz)

    result["estimated_young_gpa"] = f"{young_pa / 1e9:.6f}"
    result["closest_young_matches"] = format_matches(young_matches, "candidate", args.top_n)
    result["closest_frequency_matches"] = format_matches(frequency_matches, "candidate", args.top_n)
    result["analysis_status"] = "complete"
    return result


def main() -> None:
    args = parse_args()
    rows = list(csv.DictReader(args.metadata.open(newline="", encoding="utf-8")))
    results = [analyze_row(row, args) for row in rows]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for result in results:
        for key in result:
            if key is not None and key not in fieldnames:
                fieldnames.append(key)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: value for key, value in result.items() if key is not None} for result in results)

    print("Material study summary")
    print(f"metadata: {args.metadata}")
    print(f"results: {args.out}")
    for result in results:
        print(
            f"- {result['trial_id']}: "
            f"f={result['computed_frequency_hz'] or 'n/a'} Hz, "
            f"E={result['estimated_young_gpa'] or 'n/a'} GPa, "
            f"status={result['analysis_status']}"
        )


if __name__ == "__main__":
    main()
