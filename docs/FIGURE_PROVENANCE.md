# Proveniência das Figuras

Mapeia cada figura curada no repositório à sua origem: o script/notebook
original (em `C:\Users\rafael\PycharmProjects\pythonProject3_TCC`) ou o arquivo
de saída solto (na raiz de `C:\Users\rafael`), e indica se hoje é **reproduzível**
por um script deste repositório.

Legenda de confiança da associação: ✅ alta · 🟡 provável · ❔ incerta.

## `figures/generated/` — reproduzíveis agora

| Figura | Script deste repo |
| --- | --- |
| `signal.png`, `fft.png`, `envelope_fit.png`, `cwt_scalogram.png` | `scripts/run_basic_analysis.py` |
| `sensor/dynamics_plus_sensor_fit.png`, `sensor/sensor_output_map.png` | `scripts/run_sensor_identification.py` |

## `figures/main/` — workflow original

| Figura | Origem original | Solto em `C:\Users\rafael` | Reprodutível |
| --- | --- | --- | --- |
| `01_signal.png` | `final.ipynb` (visualização de sinal) | `Sinal.png` 🟡 | `run_basic_analysis.py` |
| `01_noisy_vs_filtered_signal.png` | `completo.py` / `wav.py` | `linear.png` ❔ | parcial |
| `02_fft.png` | `fft.py` / `final.ipynb` | `fft.png`, `fft_dado_15.png` 🟡 | `run_basic_analysis.py` |
| `03_cwt.png` | `Scalograma.py` / `recon.py` | `CWT.png` ✅ | `run_basic_analysis.py` |
| `04_hilbert_envelope.png` | `Hilbert.py` / `envelope.py` | `envelope_hilbert.png` ✅ | `run_basic_analysis.py` |
| `05_gamma_fit.png` | `envelope.py` | `Gama.png`, `ajuste_gama.png` ✅ | parcial |
| `06_dwt_decomposition.png` | `Wavelets.py` / `wav.py` | `DWT_N2.png`, `DecomposicaoWT.png` 🟡 | via `wavelet_analysis.dwt_energy` |
| `07_dominant_mode_reconstruction.png` | reconstrução DWT | `recontruscaoxoriginal.png` 🟡 | via `reconstruct_dominant_mode` |
| `08_scale_energy.png` | `Wavelets.py` | `energia_escala.png` ✅ | via `wavelet_analysis.dwt_energy` |
| `09_global_model_fit.png` | `completo.py` (otimização global) | `ajusteglobal.png` 🟡 | não (ver abaixo) |

## `figures/advanced/`

| Figura | Origem original | Solto | Reprodutível |
| --- | --- | --- | --- |
| `havok_reconstruction.png` | `recon.py` | `havok.png` ✅ | via `havok.py` |
| `sindy_identification.png` | `sindy_tratado.py` / `Synd.py` | `sindy.png` ✅ | via `sindy_model.py` / `run_advanced_analysis.py` |
| `pinn_inverse_problem.png` | `PINN.py` | `probema inverso.png` 🟡 | via `pinn.py` / `run_advanced_analysis.py --run-pinn` |
| `duffing_global_envelope_fit.png` | `prof/definitivo.ipynb` (otimização global, narrativa §11) | `ajusteglobal.png` ✅ | **sim** — `scripts/run_global_fit.py` |

## `figures/sensor/`

| Figura | Origem original | Solto | Reprodutível |
| --- | --- | --- | --- |
| `sensor_nonlinearity_fit.png` | `sss.ipynb` (output map `h(q)`) | `nonlinear.png` ✅ | via `run_sensor_identification.py` |
| `sindy_linear_model_subtraction.png` | `completo.py` (resíduo linear) | `res1.png`, `res2.png` ✅ | via `run_sensor_residual.py` |
| `sensor_residual_signature.png` | `completo.py` | — | via `run_sensor_residual.py` |
| `sensor_residual_spectrum.png` | `completo.py` | — | via `run_sensor_residual.py` |
| `sensor_residual_phase_space.png` | `completo.py` | — | via `run_sensor_residual.py` |
| `sensor_response_surface_slice.png` | `completo.py` / `model.py` | — | via `run_sensor_residual.py` |
| `sensor_residual_projection(_3d).png` | projeções SVD do resíduo | — | 🟡 variação (mesmo resíduo, outra projeção) |

## `figures/setup/` — fotos

| Figura | Origem |
| --- | --- |
| `acquisition_rig_full.jpeg`, `arduino_sensor_circuit.jpeg`, `sensor_target_alignment.jpeg` | fotos da bancada (provavelmente as `WhatsApp Image 2026-01-06 …` em `C:\Users\rafael`) ❔ |

## Status de reprodutibilidade

Todas as famílias de figuras de destaque têm agora um script reprodutível:

- **Ajuste global Duffing** (`duffing_global_envelope_fit.png`) — portado de
  `prof/definitivo.ipynb` para `global_fit.py` + `run_global_fit.py`. Reproduz a
  narrativa §11 no sample de 18 Hz: `gamma≈0.353`, `eta≈0.078`, `omega^2≈13840`
  (f≈18.72 Hz). O termo cúbico `beta` é fracamente identificável (saiu ~0; na
  narrativa oscilou 172/−755/−6.9), então o oscilador é essencialmente linear na
  rigidez, com a não-linearidade concentrada na **dissipação** (`eta`).
- **Resíduo do sensor** (`sensor_residual_*`, `sensor_response_surface_slice.png`,
  `sindy_linear_model_subtraction.png`) — portado de `completo.py` para
  `sensor_residual.py` + `run_sensor_residual.py`, **com o bug `DT=1000`
  corrigido** (ver `ORIGINAL_CODE_AUDIT.md`).

Observação: `sensor_residual_projection(_3d).png` são apenas projeções
alternativas (SVD) do mesmo resíduo já reproduzido; não foram replicadas pixel a
pixel, mas saem do mesmo `linear_dynamics_residual`.
