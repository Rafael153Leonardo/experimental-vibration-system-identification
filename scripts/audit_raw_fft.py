from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vibration_id.io import load_signal_csv
from vibration_id.preprocessing import crop_from_onset, remove_dc_offset, wavelet_denoise
from vibration_id.spectral import compute_fft, dominant_frequency, sampling_rate

SKIP_PARTS = {".git", ".idea", ".ipynb_checkpoints", ".venv", "__pycache__"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit dominant FFT frequencies for all raw CSV files in an original project tree."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Original raw/project directory to scan recursively.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "results",
        help="Output directory for audit CSV files.",
    )
    parser.add_argument("--fmin", type=float, default=1.0, help="Minimum FFT frequency in Hz.")
    parser.add_argument("--fmax", type=float, default=80.0, help="Maximum FFT frequency in Hz.")
    parser.add_argument(
        "--no-denoise",
        action="store_true",
        help="Skip wavelet denoising for the processed-frequency estimate.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit for quick checks.",
    )
    return parser.parse_args()


def should_skip(path: Path) -> bool:
    return any(part in SKIP_PARTS for part in path.parts)


def dataset_group(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    parts = rel.parts
    if len(parts) == 1:
        return "."
    if parts[0] == "Dataset_sensor":
        if len(parts) >= 3 and parts[1] == "forcado":
            return "/".join(parts[:3])
        if len(parts) >= 3 and parts[1] == "vibracao2" and parts[2] == "forcado":
            return "/".join(parts[:3])
        return "/".join(parts[:2])
    if parts[0] == "inox" and len(parts) >= 2:
        return "/".join(parts[:2])
    if parts[0] == "novo_pipeline" and len(parts) >= 2:
        return "/".join(parts[:2])
    if parts[0].lower().startswith("resultados"):
        return parts[0]
    return str(rel.parent).replace("\\", "/")


def peak_amplitude(t: np.ndarray, x: np.ndarray, frequency_hz: float) -> float:
    freqs, amp = compute_fft(t, x)
    if len(freqs) == 0:
        return float("nan")
    idx = int(np.argmin(np.abs(freqs - frequency_hz)))
    return float(amp[idx])


def analyze_file(path: Path, root: Path, args: argparse.Namespace) -> dict[str, object]:
    row: dict[str, object] = {
        "relative_path": str(path.relative_to(root)).replace("\\", "/"),
        "group": dataset_group(path, root),
        "status": "ok",
        "error": "",
        "n_samples": "",
        "duration_s": "",
        "sampling_rate_hz": "",
        "raw_frequency_hz": "",
        "raw_peak_amplitude": "",
        "processed_frequency_hz": "",
        "processed_peak_amplitude": "",
    }

    try:
        data = load_signal_csv(path, center_signal=False)
        t = data["time_s"].to_numpy(dtype=float)
        x = data["signal"].to_numpy(dtype=float)
        if len(t) < 8:
            raise ValueError("not enough samples")

        raw_x = x - np.mean(x)
        raw_frequency = dominant_frequency(t, raw_x, fmin=args.fmin, fmax=args.fmax)
        raw_amp = peak_amplitude(t, raw_x, raw_frequency)

        processed_x = remove_dc_offset(x, mode="tail")
        processed_t, processed_x, _ = crop_from_onset(t, processed_x, safety_samples=5)
        if len(processed_t) < 8:
            raise ValueError("not enough samples after onset crop")
        if not args.no_denoise:
            processed_x = wavelet_denoise(processed_x, wavelet="db8", level=2)
        processed_frequency = dominant_frequency(
            processed_t,
            processed_x,
            fmin=args.fmin,
            fmax=args.fmax,
        )
        processed_amp = peak_amplitude(processed_t, processed_x, processed_frequency)

        row.update(
            {
                "n_samples": len(t),
                "duration_s": float(t[-1] - t[0]),
                "sampling_rate_hz": sampling_rate(t),
                "raw_frequency_hz": raw_frequency,
                "raw_peak_amplitude": raw_amp,
                "processed_frequency_hz": processed_frequency,
                "processed_peak_amplitude": processed_amp,
            }
        )
    except Exception as exc:  # noqa: BLE001 - audit should keep going.
        row["status"] = "error"
        row["error"] = str(exc)
    return row


def group_summary(results: pd.DataFrame) -> pd.DataFrame:
    ok = results[results["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()

    ok["processed_frequency_hz"] = pd.to_numeric(ok["processed_frequency_hz"], errors="coerce")
    ok["raw_frequency_hz"] = pd.to_numeric(ok["raw_frequency_hz"], errors="coerce")
    ok["processed_bin_0p1_hz"] = ok["processed_frequency_hz"].round(1)

    rows: list[dict[str, object]] = []
    for group, df in ok.groupby("group", sort=True):
        counts = df["processed_bin_0p1_hz"].value_counts().head(5)
        clusters = "; ".join(f"{freq:.1f}Hz:{count}" for freq, count in counts.items())
        rows.append(
            {
                "group": group,
                "n_files": int(len(df)),
                "raw_median_hz": float(df["raw_frequency_hz"].median()),
                "processed_median_hz": float(df["processed_frequency_hz"].median()),
                "processed_mean_hz": float(df["processed_frequency_hz"].mean()),
                "processed_std_hz": float(df["processed_frequency_hz"].std(ddof=0)),
                "processed_min_hz": float(df["processed_frequency_hz"].min()),
                "processed_max_hz": float(df["processed_frequency_hz"].max()),
                "top_processed_clusters_0p1_hz": clusters,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    csv_files = sorted(path for path in root.rglob("*.csv") if not should_skip(path))
    if args.max_files is not None:
        csv_files = csv_files[: args.max_files]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(analyze_file(path, root, args) for path in csv_files)
    summary = group_summary(results)

    results_path = args.out_dir / "raw_fft_audit.csv"
    summary_path = args.out_dir / "raw_fft_group_summary.csv"
    results.to_csv(results_path, index=False, lineterminator="\n")
    summary.to_csv(summary_path, index=False, lineterminator="\n")

    print("Raw FFT audit")
    print(f"root: {root}")
    print(f"files_scanned: {len(csv_files)}")
    print(f"ok: {(results['status'] == 'ok').sum() if not results.empty else 0}")
    print(f"errors: {(results['status'] == 'error').sum() if not results.empty else 0}")
    print(f"results: {results_path}")
    print(f"summary: {summary_path}")
    if not summary.empty:
        print()
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
