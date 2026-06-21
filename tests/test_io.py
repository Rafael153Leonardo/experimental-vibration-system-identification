import numpy as np
import pandas as pd

from vibration_id.io import load_signal_csv


def _write_csv(path, df):
    df.to_csv(path, index=False)
    return path


def test_microsecond_time_is_converted_to_seconds(tmp_path):
    t_us = np.arange(0, 5000, 1000)  # 0, 1000, ..., 4000 microseconds
    df = pd.DataFrame({"tempo_us": t_us, "posicao_mm": np.sin(t_us)})
    path = _write_csv(tmp_path / "us.csv", df)

    out = load_signal_csv(path)

    # 1000 us spacing -> 0.001 s spacing
    assert np.isclose(np.median(np.diff(out["time_s"])), 0.001)
    assert out["time_s"].iloc[0] == 0.0  # center_time default


def test_seconds_columns_are_left_untouched(tmp_path):
    t = np.linspace(0.0, 1.0, 50)
    df = pd.DataFrame({"Second": t, "Volt": np.cos(t)})
    path = _write_csv(tmp_path / "sec.csv", df)

    out = load_signal_csv(path)

    assert np.isclose(np.median(np.diff(out["time_s"])), t[1] - t[0])
    assert list(out.columns) == ["time_s", "signal"]


def test_explicit_columns_override_autodetection(tmp_path):
    df = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0) ** 2})
    path = _write_csv(tmp_path / "explicit.csv", df)

    out = load_signal_csv(path, time_col="a", signal_col="b", time_unit="s")

    assert len(out) == 10
    assert np.isclose(out["signal"].iloc[-1], 81.0)


def test_non_finite_rows_are_dropped(tmp_path):
    df = pd.DataFrame({"tempo_s": [0.0, 0.1, np.nan, 0.3], "signal": [1.0, np.nan, 3.0, 4.0]})
    path = _write_csv(tmp_path / "nan.csv", df)

    out = load_signal_csv(path, time_unit="s")

    assert len(out) == 2  # only the fully-finite rows survive
