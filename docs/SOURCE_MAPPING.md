# Source Mapping

This file documents how the public repository was derived from the local project
workspace. The original folders are intentionally not copied as-is because they
contain exploratory work, IDE metadata, generated files, checkpoints, and
development history that are not useful in a public repository.

## Original Source Areas

```text
C:\Users\rafael\TCC
C:\Users\rafael\PycharmProjects\pythonProject3_TCC
C:\Users\rafael
```

## Public Repository Areas

```text
src/vibration_id/        reusable Python modules
scripts/                 reproducible entry points
data/sample/             small approved sample CSVs
figures/main/            curated baseline figures
figures/advanced/        SINDy, HAVOK and PINN figures
docs/                    repository notes and data policy
tests/                   smoke/regression tests
```

## Code Mapping

| Original file or topic | Public destination |
| --- | --- |
| `tratamento_dos_dados.py` | `src/vibration_id/io.py`, `src/vibration_id/preprocessing.py` |
| `fft.py` | `src/vibration_id/spectral.py` |
| `Hilbert.py` | `src/vibration_id/damping.py` |
| `Wavelets.py` | `src/vibration_id/wavelet_analysis.py` |
| `sindy_tratado.py`, `Synd.py`, `envelope.py` | `src/vibration_id/sindy_model.py` |
| `novo_pipeline/Hankel.py`, `novo_pipeline/SVD.py`, `novo_pipeline/HAVOK.py` | `src/vibration_id/havok.py` |
| `novo_pipeline/Pinn.py` | `src/vibration_id/pinn.py`, `scripts/run_advanced_analysis.py` |
| exploratory notebooks | `notebooks/README.md` with a recommended notebook sequence |

## Data Mapping

Curated real CSV files are included as public samples. Synthetic data is kept as
an additional reproducibility option.

| Original file | Public sample |
| --- | --- |
| `C:\Users\rafael\TCC\dados_perfeitos.csv` | `data/sample/sample_vibration_18hz.csv` |
| `Dataset_sensor\inox\dados_calibrados_01.csv` | `data/sample/sample_inox_raw_calibrated.csv` |
| `Resultados_Finais_Sincronizados\sinc_dados_calibrados_15.csv` | `data/sample/sample_inox_synchronized.csv` |
| `novo_pipeline\relatorio_ruido_tcc.csv` | `data/sample/noise_summary.csv` |
| synthetic damped oscillator | `data/synthetic/synthetic_damped_oscillator.csv` |

## Figure Mapping

| Original figure | Public figure |
| --- | --- |
| `Sinal.png` | `figures/main/01_signal.png` |
| `fft.png` | `figures/main/02_fft.png` |
| `CWT.png` | `figures/main/03_cwt.png` |
| `envelope_hilbert.png` | `figures/main/04_hilbert_envelope.png` |
| `ajuste_gama.png` | `figures/main/05_gamma_fit.png` |
| `DecomposicaoWT.png` | `figures/main/06_dwt_decomposition.png` |
| `recontruscaoxoriginal.png` | `figures/main/07_dominant_mode_reconstruction.png` |
| `energia_escala.png` | `figures/main/08_scale_energy.png` |
| `ajusteglobal.png` | `figures/main/09_global_model_fit.png` |
| `sindy.png` | `figures/advanced/sindy_identification.png` |
| `havok.png` | `figures/advanced/havok_reconstruction.png` |
| `probema inverso.png` | `figures/advanced/pinn_inverse_problem.png` |

## Publication Notes

- Keep full raw datasets outside the repository unless there is explicit
  permission to publish them.
- Prefer scripts and modules over notebooks with long duplicated cells.
- Commit generated figures only when they are curated and relevant to the
  project story.
- Use GitHub releases, Zenodo, Google Drive, or an institutional repository for
  large data if public redistribution is allowed.
