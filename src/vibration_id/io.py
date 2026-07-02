from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TIME_CANDIDATES = (
    "time_s",
    "tempo_s",
    "second",
    "Second",
    "tempo_corrigido",
    "tempo_fixed_str",
    "tempo_num",
    "x-axis",
    "t",
    "tempo",
    "tempo_us",
)

SIGNAL_CANDIDATES = (
    "signal",
    "x",
    "x_svd_limpo",
    "x_original",
    "posicao_mm_cent",
    "posicao_mm",
    "Volt",
    "volt",
    "volt_num",
    "volt_fixed_str",
    "1",
    "amplitude",
)


def _find_column(columns: list[str], candidates: tuple[str, ...], explicit: str | None) -> str:
    if explicit is not None:
        if explicit not in columns:
            raise ValueError(f"Column '{explicit}' not found. Available columns: {columns}")
        return explicit

    for name in candidates:
        if name in columns:
            return name

    raise ValueError(f"No matching column found. Available columns: {columns}")


def load_signal_csv(
    path: str | Path,
    *,
    time_col: str | None = None,
    signal_col: str | None = None,
    time_unit: str = "auto",
    center_time: bool = True,
    center_signal: bool = False,
) -> pd.DataFrame:
    """Load a vibration CSV and normalize it to columns time_s and signal.

    Supported project formats include:
    - tempo_us, posicao_mm
    - tempo_s, posicao_mm_cent
    - tempo_corrigido, volt
    - Second, Volt
    """

    path = Path(path)
    df = pd.read_csv(path)

    columns = list(df.columns)
    t_name = _find_column(columns, TIME_CANDIDATES, time_col)
    x_name = _find_column(columns, SIGNAL_CANDIDATES, signal_col)

    t = pd.to_numeric(df[t_name], errors="coerce").to_numpy(dtype=float)
    x = pd.to_numeric(df[x_name], errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(t) & np.isfinite(x)
    t = t[mask]
    x = x[mask]

    if time_unit == "auto":
        if "us" in t_name.lower():
            dt = np.nanmedian(np.diff(np.sort(t))) if len(t) > 1 else 0.0
            if np.nanmax(np.abs(t)) > 10_000.0 or dt > 10.0:
                t = t / 1_000_000.0
        elif len(t) > 1:
            dt = np.nanmedian(np.diff(np.sort(t)))
            if dt > 10.0:
                t = t / 1_000_000.0
    elif time_unit in {"us", "microsecond", "microseconds"}:
        t = t / 1_000_000.0
    elif time_unit not in {"s", "second", "seconds"}:
        raise ValueError("time_unit must be 'auto', 's', or 'us'")

    order = np.argsort(t)
    t = t[order]
    x = x[order]

    if center_time and len(t):
        t = t - t[0]
    if center_signal and len(x):
        x = x - np.nanmean(x)

    return pd.DataFrame({"time_s": t, "signal": x})


def save_signal_csv(data: pd.DataFrame, path: str | Path) -> None:
    """Save normalized time series data."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, lineterminator="\n")
