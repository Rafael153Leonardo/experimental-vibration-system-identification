from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vibration_id.damping import fit_exponential_envelope, quality_factor
from vibration_id.pipeline import load_clean_signal
from vibration_id.plotting import (
    save_envelope_plot,
    save_fft_plot,
    save_noisy_vs_filtered_plot,
    save_scalogram_plot,
    save_signal_plot,
)
from vibration_id.preprocessing import crop_from_onset, remove_dc_offset, wavelet_denoise
from vibration_id.spectral import compute_fft, dominant_frequency
from vibration_id.synthetic import damped_oscillator
from vibration_id.wavelet_analysis import cwt_scalogram


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the baseline vibration analysis pipeline.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "data" / "sample" / "sample_vibration_18hz.csv",
        help="Input CSV. If missing, a synthetic signal is used.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "figures" / "generated",
        help="Output directory for generated figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.csv.exists():
        cleaned = load_clean_signal(args.csv)
        t, x_raw, x_clean, onset = cleaned.t, cleaned.raw, cleaned.clean, cleaned.onset
        source = str(args.csv)
    else:
        t, x_raw = damped_oscillator()
        x_raw = remove_dc_offset(x_raw, mode="tail")
        t, x_raw, onset = crop_from_onset(t, x_raw, safety_samples=5)
        x_clean = wavelet_denoise(x_raw, wavelet="db8", level=2)
        source = "synthetic"

    f0 = dominant_frequency(t, x_clean, fmin=1.0, fmax=80.0)
    freqs, amp = compute_fft(t, x_clean)
    damping = fit_exponential_envelope(t, x_clean, tmin=0.05, tmax=min(15.0, float(t[-1]) * 0.5))
    q_factor = quality_factor(f0, damping.gamma)
    cwt_freqs, _, power = cwt_scalogram(t, x_clean, fmin=1.0, fmax=80.0)

    save_signal_plot(t, x_clean, args.out / "signal.png", title="Clean vibration signal")
    save_noisy_vs_filtered_plot(t, x_raw, x_clean, args.out / "noisy_vs_filtered.png")
    save_fft_plot(freqs, amp, args.out / "fft.png", fmax=80.0)
    save_envelope_plot(t, x_clean, damping.envelope, damping.model, args.out / "envelope_fit.png")
    save_scalogram_plot(t, cwt_freqs, power, args.out / "cwt_scalogram.png")

    print("Baseline vibration analysis")
    print(f"source: {source}")
    print(f"onset_index: {onset}")
    print(f"dominant_frequency_hz: {f0:.4f}")
    print(f"gamma: {damping.gamma:.6f}")
    print(f"Q: {q_factor:.4f}")
    print(f"envelope_r_squared: {damping.r_squared:.4f}")
    print(f"figures: {args.out}")


if __name__ == "__main__":
    main()
