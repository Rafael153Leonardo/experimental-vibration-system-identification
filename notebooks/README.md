# Notebooks

This folder is intentionally light. The reproducible analyses live as
command-line scripts in `scripts/`, each calling the reusable functions in
`src/vibration_id`. There are no committed notebooks yet; the table below maps
the originally planned notebook story to the script that already covers it, so
the work is reproducible without large exploratory cells.

| Planned notebook | Reproducible script today |
| --- | --- |
| `01_data_cleaning_and_calibration` | `scripts/run_basic_analysis.py` (load, onset, denoise) |
| `02_fft_psd_analysis` | `scripts/run_basic_analysis.py` (FFT) |
| `03_wavelet_hilbert_damping` | `scripts/run_basic_analysis.py` (CWT, Hilbert envelope) |
| `04_sindy_system_identification` | `scripts/run_advanced_analysis.py` (SINDy) |
| `05_havok_svd_advanced` | `scripts/run_advanced_analysis.py` (HAVOK) |
| `06_pinn_optional` | `scripts/run_advanced_analysis.py --run-pinn` |
| sensor + Duffing identification | `scripts/run_sensor_identification.py` |
| global Duffing fit | `scripts/run_global_fit.py` |
| sensor residual analysis | `scripts/run_sensor_residual.py` |

If notebooks are added later, they should import from `src/vibration_id`
instead of redefining long code blocks, and stay output-light.
