# Figure Provenance

Maps each curated figure in the repo to its origin: the original script/notebook
(in `pythonProject3_TCC`) or the loose output file (in the home folder), and
notes whether it is **reproducible** today by a script in this repo.

Association-confidence legend: ✅ high · 🟡 likely · ❔ uncertain.

**Update (July 2026):** after the identification-pipeline fixes, the curated
figures marked *regenerated 2026-07* below were re-exported by the repo scripts
(English titles, corrected linear/Duffing/PINN identification). The "Original
source" column still records where each analysis came from; the original loose
files remain the historical reference.

## `figures/generated/` — reproducible now

| Figure | Script in this repo |
| --- | --- |
| `signal.png`, `fft.png`, `envelope_fit.png`, `cwt_scalogram.png` | `scripts/run_basic_analysis.py` |
| `sensor/dynamics_plus_sensor_fit.png`, `sensor/sensor_output_map.png` | `scripts/run_sensor_identification.py` |
| `sensor/sensor_nonlinearity_fit.png`, `sensor/sensor_residual_*.png` | `scripts/run_sensor_residual.py` |

## `figures/main/` — original workflow

| Figure | Original source | Loose file | Reproducible |
| --- | --- | --- | --- |
| `01_signal.png` | `final.ipynb` (signal view) | `Sinal.png` 🟡 | `run_basic_analysis.py` |
| `01_noisy_vs_filtered_signal.png` | `completo.py` / `wav.py` | `linear.png` ❔ | **yes** — `run_basic_analysis.py` (regenerated 2026-07) |
| `02_fft.png` | `fft.py` / `final.ipynb` | `fft.png`, `fft_dado_15.png` 🟡 | **yes** — `run_basic_analysis.py` (regenerated 2026-07) |
| `03_cwt.png` | `Scalograma.py` / `recon.py` | `CWT.png` ✅ | `run_basic_analysis.py` |
| `04_hilbert_envelope.png` | `Hilbert.py` / `envelope.py` | `envelope_hilbert.png` ✅ | `run_basic_analysis.py` |
| `05_gamma_fit.png` | `envelope.py` | `Gama.png`, `ajuste_gama.png` ✅ | partial |
| `06_dwt_decomposition.png` | `Wavelets.py` / `wav.py` | `DWT_N2.png`, `DecomposicaoWT.png` 🟡 | via `wavelet_analysis.dwt_energy` |
| `07_dominant_mode_reconstruction.png` | DWT reconstruction | `recontruscaoxoriginal.png` 🟡 | via `reconstruct_dominant_mode` |
| `08_scale_energy.png` | `Wavelets.py` | `energia_escala.png` ✅ | via `wavelet_analysis.dwt_energy` |
| `09_global_model_fit.png` | `completo.py` (global optimization) | `ajusteglobal.png` 🟡 | no (see below) |

## `figures/advanced/`

| Figure | Original source | Loose file | Reproducible |
| --- | --- | --- | --- |
| `havok_reconstruction.png` | `recon.py` | `havok.png` ✅ | **yes** — `run_advanced_analysis.py` (regenerated 2026-07) |
| `sindy_identification.png` | `sindy_tratado.py` / `Synd.py` | `sindy.png` ✅ | **yes** — `run_advanced_analysis.py` (regenerated 2026-07) |
| `pinn_inverse_problem.png` | `PINN.py` / `novo_pipeline/Pinn.py` | `probema inverso.png` 🟡 | **yes** — `run_advanced_analysis.py --run-pinn` (regenerated 2026-07; now the two-stage fit on the inox sample with parameter convergence) |
| `duffing_global_envelope_fit.png` | `prof/definitivo.ipynb` (global optimization, narrative §11) | `ajusteglobal.png` ✅ | **yes** — `scripts/run_global_fit.py` |
| `backbone_damping.png` | new analysis (2026-07), not in the original tree | — | **yes** — `scripts/run_backbone_damping.py` (backbone curve + damping-law comparison on the inox sample) |
| `digital_twin.png` | new analysis (2026-07), not in the original tree | — | **yes** — `scripts/run_digital_twin.py` (validated forward model + four virtual experiments) |

## `figures/sensor/`

| Figure | Original source | Loose file | Reproducible |
| --- | --- | --- | --- |
| `sensor_nonlinearity_fit.png` | `testes.py` (position residual vs simulated linear state, cubic fit) | `nonlinear.png` ✅ | **yes** — `run_sensor_residual.py` (regenerated 2026-07) |
| `sindy_linear_model_subtraction.png` | `completo.py` / `testes.py` (linear residual) | `res1.png`, `res2.png` ✅ | **yes** — `run_sensor_residual.py` (regenerated 2026-07) |
| `sensor_residual_signature.png` | `completo.py` | — | **yes** — `run_sensor_residual.py` (regenerated 2026-07) |
| `sensor_residual_spectrum.png` | `completo.py` | — | **yes** — `run_sensor_residual.py` (regenerated 2026-07) |
| `sensor_residual_phase_space.png` | `completo.py` | — | **yes** — `run_sensor_residual.py` (regenerated 2026-07) |
| `sensor_response_surface_slice.png` | `completo.py` / `model.py` | — | **yes** — `run_sensor_residual.py` (regenerated 2026-07) |
| `sensor_residual_projection(_3d).png` | SVD projections of the residual | — | 🟡 variation (same residual, different projection) |

## `figures/story/` — narrative figures (July 2026)

| Figure | Source |
| --- | --- |
| `01_measurement.png`, `cover_signal_to_physics.gif` | generated from `data/sample/sample_inox_raw_calibrated.csv` by `scripts/make_story_figures.py` / `scripts/make_cover_gif.py` |
| `02_modal_ladder.png` | ideal ladder from `beam_modes.py`; measured forced-mode frequencies as documented in `ORIGINAL_CODE_AUDIT.md` (raw forced dataset is not public) |
| `03_verdict.png` | Young-modulus ladder recomputed at plot time by `materials.young_modulus_from_frequency` (f₁ = 4.982 Hz, L = 0.300 m, b = 25 mm, ρ = 7850 kg/m³, 0.21 g tip mass; h = 1.5 / 1.0 / 0.55 mm → 27 / 61 / 205.3 GPa) + the uncertainty budget (±2.6% total, thickness-dominated). The 17.6–18 GPa quoted by the original exploratory analysis additionally baked in an effective length L = 0.270 m and is **not** reproduced by the pipeline (see `ORIGINAL_CODE_AUDIT.md`) |

## `figures/setup/` — photos

| Figure | Origin |
| --- | --- |
| `acquisition_rig_full.jpeg`, `arduino_sensor_circuit.jpeg`, `sensor_target_alignment.jpeg` | bench photos (likely the `WhatsApp Image 2026-01-06 …` files in the home folder) ❔ |

## Reproducibility status

Every headline figure family now has a reproducible script:

- **Global Duffing fit** (`duffing_global_envelope_fit.png`) — ported from
  `prof/definitivo.ipynb` into `global_fit.py` + `run_global_fit.py`. Reproduces
  narrative §11 on the 18 Hz sample: `gamma≈0.353`, `eta≈0.078`, `omega^2≈13840`
  (f≈18.72 Hz). The cubic term `beta` is weakly identifiable (came out ~0; it
  swung 172/−755/−6.9 in the narrative), so the oscillator is essentially linear
  in stiffness, with the nonlinearity concentrated in the **dissipation** (`eta`).
- **Backbone + damping law** (`backbone_damping.png`) — new in `backbone.py` +
  `run_backbone_damping.py`. Reads the instantaneous frequency and amplitude off
  the analytic signal of the inox ring-down and shows the two facts the global
  fit only hinted at, directly from the data: the **backbone is flat**
  (frequency varies <1% across a threefold amplitude decay, so the cubic
  stiffness term is a sensor-map artifact, not beam physics), and the ring-down
  **envelope prefers amplitude-dependent damping** over a single exponential
  (R²≈0.95→0.998 on the sample, Q≈120). This localizes the nonlinearity: linear
  stiffness, nonlinear dissipation.
- **Digital twin** (`digital_twin.png`) — new in `digital_twin.py` +
  `run_digital_twin.py`. Composes the identified pieces (Euler-Bernoulli modal
  frequencies, nonlinear damping, cubic sensor map) into one forward model and
  validates it against the real inox sample: the synthetic ring-down, run through
  the same pipeline, recovers f₀ = 4.982 Hz (to 4 digits), Q and the R² damping
  signature, the βₙ² modal ladder and the steel verdict. The same figure runs the
  four virtual experiments (pluck, forced sweep, geometry what-if, tip-mass
  sensing). An interactive version is in `interactive/digital_twin.html`.
- **Sensor residual** (`sensor_residual_*`, `sensor_response_surface_slice.png`,
  `sindy_linear_model_subtraction.png`) — ported from `completo.py` into
  `sensor_residual.py` + `run_sensor_residual.py`, **with the `DT=1000` bug
  fixed** (see `ORIGINAL_CODE_AUDIT.md`). The linear model is identified on a
  contiguous full-rate window: the strided decimation used in an earlier port
  jittered the time grid and biased the frequency ~20% low (4.04 vs 4.98 Hz).
- **Static sensor nonlinearity** (`sensor_nonlinearity_fit.png`) — originally
  produced by `testes.py` (not `sss.ipynb` as previously recorded): simulate
  the identified linear model, take the position residual
  `measured − simulated` and fit a cubic against the simulated state. Ported as
  `simulate_linear_state` + `fit_static_nonlinearity` in `sensor_residual.py`,
  regenerated by `run_sensor_residual.py`.

Note: `sensor_residual_projection(_3d).png` are just alternative (SVD)
projections of the same already-reproduced residual; they were not replicated
pixel-for-pixel, but come from the same `linear_dynamics_residual`.
