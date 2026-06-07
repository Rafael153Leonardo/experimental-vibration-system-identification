# Project Structure

The original research folder contained exploratory scripts, notebooks, raw data,
processed data, and generated figures in the same tree. This public version is
split into stable project areas.

## Core Modules

```text
io.py                 CSV loading and normalization
preprocessing.py      offset removal, onset detection, denoising, velocity
spectral.py           FFT, PSD, dominant frequency
damping.py            Hilbert envelope and exponential damping fit
wavelet_analysis.py   CWT, DWT energy, dominant mode reconstruction
sindy_model.py        optional SINDy oscillator identification
havok.py              Hankel matrix, SVD and HAVOK identification
synthetic.py          synthetic damped oscillator generation
plotting.py           figure writers used by scripts
```

## Recommended Notebook Story

```text
01_data_cleaning_and_calibration.ipynb
02_fft_psd_analysis.ipynb
03_wavelet_hilbert_damping.ipynb
04_sindy_system_identification.ipynb
05_havok_svd_advanced.ipynb
06_pinn_optional.ipynb
```

Keep notebooks output-light and call functions from `src/vibration_id` instead
of redefining long code blocks in each notebook.

