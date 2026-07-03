"""Generate the README cover GIF: physics emerging from a low-cost sensor.

Left panel: the inox ruler's free decay draws itself in time (oscilloscope
sweep). Right panel: the FFT of the data seen so far — the 4.982 Hz line
sharpens as the record grows. The whole repo in one loop: raw signal in,
physical number out.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vibration_id.pipeline import decimate, load_clean_signal

STORY = ROOT / "figures" / "story"

SWEEP_SECONDS = 12.0
N_FRAMES = 80
FPS = 18
HOLD_FRAMES = 20


def main() -> None:
    cleaned = load_clean_signal(ROOT / "data" / "sample" / "sample_inox_raw_calibrated.csv")
    t, x = decimate(cleaned.t, cleaned.clean, 6000)
    mask = t <= SWEEP_SECONDS
    t, x = t[mask], x[mask]
    dt = float(np.median(np.diff(t)))

    fig, (ax_time, ax_fft) = plt.subplots(1, 2, figsize=(9.6, 3.6), dpi=100, gridspec_kw={"width_ratios": [3, 2]})
    y_max = 1.1 * float(np.max(np.abs(x)))
    ax_time.set_xlim(0.0, SWEEP_SECONDS)
    ax_time.set_ylim(-y_max, y_max)
    ax_time.set_xlabel("Time [s]")
    ax_time.set_ylabel("Position [mm]")
    ax_time.set_title("A steel ruler, a $2 sensor, an Arduino...")
    ax_time.grid(True, alpha=0.3)
    (line_signal,) = ax_time.plot([], [], color="gray", linewidth=0.5)

    ax_fft.set_xlim(0.0, 12.0)
    ax_fft.set_ylim(0.0, 1.12)
    ax_fft.set_xlabel("Frequency [Hz]")
    ax_fft.set_ylabel("FFT (normalized)")
    ax_fft.set_title("...and physics emerging: 4.982 Hz")
    ax_fft.grid(True, alpha=0.3)
    (line_fft,) = ax_fft.plot([], [], color="crimson", linewidth=1.2)
    fig.tight_layout()

    cut_indices = np.linspace(int(0.4 / dt), len(t) - 1, N_FRAMES).astype(int)
    frames: list[Image.Image] = []
    for cut in cut_indices:
        line_signal.set_data(t[:cut], x[:cut])
        segment = x[:cut] - np.mean(x[:cut])
        amp = np.abs(np.fft.rfft(segment * np.hanning(len(segment))))
        freqs = np.fft.rfftfreq(len(segment), d=dt)
        band = freqs <= 12.0
        line_fft.set_data(freqs[band], amp[band] / float(np.max(amp[band])))
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(Image.fromarray(rgba[..., :3]).convert("P", palette=Image.Palette.ADAPTIVE))
    plt.close(fig)

    frame_ms = int(round(1000.0 / FPS))
    durations = [frame_ms] * (len(frames) - 1) + [frame_ms * HOLD_FRAMES]
    STORY.mkdir(parents=True, exist_ok=True)
    out = STORY / "cover_signal_to_physics.gif"
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    print(f"gif: {out} ({out.stat().st_size / 1e6:.2f} MB, {len(frames)} frames @ {FPS} fps + hold)")


if __name__ == "__main__":
    main()
