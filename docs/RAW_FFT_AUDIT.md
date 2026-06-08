# Raw FFT Audit

This audit was run against the original local project tree:

```text
C:\Users\rafael\PycharmProjects\pythonProject3_TCC
```

Command:

```text
python scripts\audit_raw_fft.py --root C:\Users\rafael\PycharmProjects\pythonProject3_TCC --fmin 1 --fmax 80 --no-denoise
```

The script scanned `419` CSV files and successfully computed FFT peaks for
`411` files. The remaining files are non-time-series reports, unrelated radar
tables, malformed CSV exports, or records too short after onset cropping.

## Main Frequency Groups

| Group | Files | Median processed frequency |
| --- | ---: | ---: |
| `Dataset_sensor/inox` | 29 | 4.982 Hz |
| `Resultados_Centralizados` | 29 | 4.982 Hz |
| `Resultados_Sincronizados` | 28 | 4.982 Hz |
| `Resultados_Finais_Sincronizados` | 21 | 4.982 Hz |
| `inox/dados` | 10 | 4.984 Hz |
| `resultados_limpos` | 10 | 4.986 Hz |
| `Dataset_sensor/vibracao` | 21 | 5.119 Hz |
| `inox_centralizado` | 29 | 5.110 Hz |
| `Dataset_sensor/vibracao2` | 39 | 11.278 Hz |
| `prof` / treated professor data | multiple | ~18.76 Hz |
| `Dataset_sensor/forcado/segundo_harmonico` | 5 | 31.194 Hz |
| `inox/ruido` | 9 | 59.996 Hz |

## Naming Finding

The raw inox datasets are internally consistent. `Dataset_sensor/inox`,
`Resultados_Centralizados`, `Resultados_Sincronizados`,
`Resultados_Finais_Sincronizados`, `inox/dados`, `resultados_limpos`, and the
new-pipeline synchronized outputs all cluster around `5 Hz`.

The likely naming problem is `inox_centralizado`. The original script
`centralizar_dados.py` reads:

```text
Dataset_sensor/vibracao
```

but writes:

```text
inox_centralizado/cent_exp_*.csv
```

So `inox_centralizado` should not be treated as a clean inox-derived folder
without checking each file. Its FFT cluster mostly follows `Dataset_sensor/vibracao`
around `5.1 Hz`, with a few files near `5.0 Hz` and a small number of outliers.

## Interpretation

The FFT audit supports the original experimental observation: the inox-related
raw and processed datasets consistently identify a natural frequency near
`4.98 Hz`. The discrepancy found earlier did not come from inconsistent FFT
extraction; it came from incomplete or mixed metadata about the experimental
geometry, tip paper target, and folder lineage.

Detailed outputs:

```text
data/results/raw_fft_audit.csv
data/results/raw_fft_group_summary.csv
```
