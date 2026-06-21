from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vibration_id.synthetic import damped_oscillator


def main() -> None:
    out = ROOT / "data" / "synthetic" / "synthetic_damped_oscillator.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    t, x = damped_oscillator()
    pd.DataFrame({"time_s": t, "signal": x}).to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()
