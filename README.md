# Experimental Vibration System Identification

[![CI](https://github.com/Rafael153Leonardo/experimental-vibration-system-identification/actions/workflows/ci.yml/badge.svg)](https://github.com/Rafael153Leonardo/experimental-vibration-system-identification/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)

**From a raw optical-sensor signal to interpretable physics** — a Python pipeline
that takes vibration measurements off a homemade rig and recovers natural
frequencies, damping, dynamic models, and even a material's Young modulus. It
combines classical signal processing, data-driven system identification
(SINDy / HAVOK / PINN), and physics-based beam modeling.

I built the bench, adapted the sensor, acquired the data with an Arduino, and
wrote the full analysis pipeline. This repo is the cleaned, tested, reproducible
version of that work.

| Noisy vs filtered signal | FFT spectrum | Global physical fit |
| --- | --- | --- |
| ![Noisy vs filtered signal](figures/main/01_noisy_vs_filtered_signal.png) | ![FFT](figures/main/02_fft.png) | ![Global physical fit](figures/advanced/duffing_global_envelope_fit.png) |

## Highlight — measuring a steel ruler's Young modulus from how it vibrates

A stainless-steel ruler in free vibration gave a fundamental of `4.98 Hz`.
Feeding that into the Euler–Bernoulli cantilever model returned `~18 GPa` — an
order of magnitude below steel. Instead of fudging the number, I designed a
**forced-vibration experiment**, driving the same ruler at its first four modes
(`~5.0 / 31.2 / 86.8 / 173.5 Hz`).

Cantilever modes are *not* integer harmonics — they follow the `beta_n^2` ladder
(`1 : 6.27 : 17.5 : 34.4`), and that ladder is a fingerprint of the boundary
condition, independent of the material. The measured ratios matched the ideal
ladder within **~1%**, which proves the clamp was essentially perfect and rules
it out as the cause (a soft clamp or a tip mass would push the higher-mode ratios
*up*, not onto the ideal line). That left geometry: a micrometer read the blade
at `0.55 mm` (the `1 mm` marking is nominal). With the true thickness, the inverse
gives `E ≈ 205 GPa` — squarely stainless / carbon steel.

The takeaway, and the reusable code in [`beam_modes.py`](src/vibration_id/beam_modes.py):
**higher modes let you separate the boundary condition from the material.** The
full investigation is in [`docs/ORIGINAL_CODE_AUDIT.md`](docs/ORIGINAL_CODE_AUDIT.md).

## What's inside

**Signal processing & spectral analysis**
- Windowed FFT with sub-bin parabolic peak interpolation; Welch PSD
- Wavelets: CWT scalograms, DWT multiresolution energy, wavelet denoising
- Hilbert-envelope damping fits; physical quality factor `Q = 2*pi*f0/gamma` and a half-power-bandwidth estimator

**Data-driven system identification**
- SINDy — linear oscillator and free-Duffing models (with a dependency-free STLSQ)
- HAVOK (Hankel + SVD + regression) and ensemble-SVD reconstruction
- Physics-Informed Neural Network (PyTorch) with trainable physical parameters
- 3-stage global Duffing fit: nonlinear envelope → phase calibration → ODE optimization
- Sensor output map `h(q)` and linear-model residual analysis for the sensor nonlinearity

**Physics-based modeling**
- Euler–Bernoulli cantilever (forward + inverse) with a Rayleigh tip-mass correction
- Modal-ladder analysis to separate clamp quality from material modulus

**Hardware & engineering**
- Modified TCRT5000 reflective optical sensor + Arduino Uno (~1 kHz), raw streaming over serial
- Typed, modular package · 39 regression tests · ruff lint + format · GitHub Actions CI (3.10 / 3.12 + a job with the pysindy/torch extras)

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .
python scripts/run_basic_analysis.py
```

`pip install -e .` puts `vibration_id` on the path; the scripts also add `src/`
to `sys.path`, so they run from a clean checkout without installing. Generated
figures land in `figures/generated/`.

```bash
# tests
pip install -r requirements-dev.txt && pytest -q

# advanced methods (SINDy / HAVOK / PINN; the PINN demo trains on the inox sample)
pip install -r requirements-advanced.txt
python scripts/run_advanced_analysis.py --run-pinn

# reproducible Duffing + sensor identification and the global fit
python scripts/run_sensor_identification.py
python scripts/run_global_fit.py
python scripts/run_sensor_residual.py

# material study from trial metadata
python scripts/run_material_study.py
```

## Results

| Context | Frequency | Geometry | Young modulus | Verdict |
| --- | ---: | --- | ---: | --- |
| Plastic ruler | `7.060 Hz` | `L=0.300 m, h=2.33 mm` | `2.99 GPa` | acrylic / PVC / polystyrene range |
| Inox ruler (raw) | `4.982 Hz` | `L=0.300 m, h=0.55 mm` | `205.3 GPa` | **stainless / carbon steel** (190–210 GPa) |
| Inox ruler (synchronized) | `4.982 Hz` | `L=0.300 m, h=0.55 mm` | `205.3 GPa` | reproduces the raw inox result |
| 18 Hz baseline | `18.737 Hz` | not documented | n/a | signal-analysis validation only |

| Sensor nonlinearity | HAVOK reconstruction | PINN inverse problem |
| --- | --- | --- |
| ![Sensor nonlinearity](figures/sensor/sensor_nonlinearity_fit.png) | ![HAVOK](figures/advanced/havok_reconstruction.png) | ![PINN](figures/advanced/pinn_inverse_problem.png) |

## Hardware and acquisition

Measurements were acquired with a modified TCRT5000 reflective optical sensor on
an Arduino Uno, sampling the raw analog signal at `~1000 Hz` and streaming
`tempo_us,leitura_bruta` over serial; all calibration and modeling happen later
in Python. The Arduino sketch is in [`hardware/arduino/AMM.ino`](hardware/arduino/AMM.ino).

| Experimental rig | Sensor circuit | Sensor alignment |
| --- | --- | --- |
| ![Experimental rig](figures/setup/acquisition_rig_full.jpeg) | ![Arduino sensor circuit](figures/setup/arduino_sensor_circuit.jpeg) | ![Sensor alignment](figures/setup/sensor_target_alignment.jpeg) |

## Baseline pipeline

1. Load a CSV and normalize columns to `time_s` and `signal`.
2. Remove the DC offset and crop to the vibration onset.
3. Wavelet-denoise.
4. Estimate the dominant frequency (windowed FFT, sub-bin interpolation).
5. Fit the Hilbert envelope and damping; compute the quality factor.
6. Generate a CWT scalogram.
7. Estimate the material class with the Euler–Bernoulli model.

The loader accepts the project's CSV variants
(`tempo_us,posicao_mm` · `tempo_s,posicao_mm_cent` · `tempo_corrigido,volt` ·
`Second,Volt`) and normalizes everything to `time_s,signal`.

## Documentation

- [`docs/EXPERIMENTAL_NARRATIVE.md`](docs/EXPERIMENTAL_NARRATIVE.md) — hardware, sensor calibration and the identification workflow
- [`docs/ORIGINAL_CODE_AUDIT.md`](docs/ORIGINAL_CODE_AUDIT.md) — audit of the original exploratory code and how it was ported / corrected here
- [`docs/FIGURE_PROVENANCE.md`](docs/FIGURE_PROVENANCE.md) — figure-by-figure origin and reproducibility
- [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md) — module map

## License

Released under the MIT License. See [`LICENSE`](LICENSE).
