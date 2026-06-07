# Experimental Narrative: testes.py and definitivo.ipynb

This note explains the role of two original files in the project:

```text
C:\Users\rafael\AppData\Roaming\JetBrains\PyCharm2025.2\scratches\testes.py
C:\Users\rafael\PycharmProjects\pythonProject3_TCC\prof\definitivo.ipynb
```

## Short Narrative

The project starts as a home-built vibration experiment and evolves into a
complete system-identification workflow. The objective is not only to plot a
decaying oscillation, but to transform raw measurements into a physical model
with interpretable parameters: natural frequency, damping, nonlinear energy
dissipation and possible Duffing stiffness.

The exploratory script `testes.py` is a small Matplotlib scratch file. It builds
a two-dimensional grid, evaluates the scalar field `Z = 1 / (X^2 + Y^2)` and
draws contour lines. This script is not the core vibration pipeline; its value is
as a plotting prototype. It tests how contour levels behave around a strong
central singularity and can be interpreted as a visual experiment for scalar
fields, energy landscapes or cost-function surfaces.

The notebook `definitivo.ipynb` is the main technical narrative. It connects the
experimental data to a sequence of increasingly structured models: first the
signals are cleaned and synchronized, then SVD is used to remove noise, then
SINDy is used to discover candidate governing equations, and finally a
phenomenological oscillator is fitted by numerical optimization.

## Role of testes.py

`testes.py` performs four operations:

1. Imports `matplotlib` and `numpy`.
2. Creates a dense `300 x 300` grid in the interval `[-1, 1]`.
3. Computes `Z = 1 / (X^2 + Y^2)`.
4. Draws contour lines with `ax.contour`.

Technically, this is a visualization test rather than an identification script.
The function has a singularity at the origin, so the contour plot concentrates
high values near the center and produces wider bands farther away. This makes it
useful as a quick check of Matplotlib contour behavior, but it should not be
presented as evidence for the vibration model.

The best way to describe it in the project is:

```text
Exploratory contour-plot scratch used to test scalar-field visualization before
building richer figures for the vibration analysis.
```

The figure generated from this scratch is included as:

![Contour scalar field test](../figures/advanced/contour_scalar_field_test.png)

## Role of definitivo.ipynb

`definitivo.ipynb` is the final experimental notebook. It has 22 cells and
organizes the work into these phases.

### 1. Raw Data Preparation

The notebook starts by loading synchronized oscilloscope-style CSV files and
detecting the start of the oscillation by the largest derivative jump. In the
captured execution, 9 files were processed and the oscillation was shifted to
start at `t = 0`.

Next, each signal is corrected:

- time is zeroed;
- signal offset is removed;
- corrected files are saved into a dedicated folder.

The notebook reports 9 corrected files, including `sinc_sinc_scope_04.csv`
through `sinc_sinc_scope_10.csv` and two validation files.

### 2. Visual Inspection

The corrected signals are plotted in grouped figures. This stage is important
because it checks whether the preprocessing produced comparable trajectories
before any model is trained or fitted.

### 3. SVD Denoising

The notebook then applies a Hankel/SVD-style reconstruction. The core function
builds a windowed trajectory matrix, decomposes it with SVD and reconstructs the
signal using selected components.

The execution used:

```text
window L = 100
components kept = [0, 1]
```

The estimated removed noise was approximately:

```text
0.0401 to 0.0467
```

This creates the column `x_svd_limpo`, which becomes the cleaner input for the
identification steps.

### 4. SINDy and Duffing Discovery

After SVD cleaning, the notebook prepares multiple trajectories for PySINDy. It
constructs state vectors from position and velocity and fits sparse polynomial
models.

The first relevant discovery is a Duffing-style candidate:

```text
(x)' = 1.000 v
(v)' = -20.285 -13892.495 x -0.508 v + 1.132 x^2 -0.214 x v + 4.082 x^3
```

Two SINDy configurations are then compared:

```text
Model A R2 = 0.99973
Model B R2 = 0.99974
```

The notebook concludes that the cubic stiffness term is small compared with the
linear stiffness term. The practical interpretation is that the experiment is
predominantly a linear-stiffness oscillator, while nonlinear effects appear more
clearly in the damping/energy decay than in the spring term.

### 5. Smoothed Derivatives

Because velocity estimates are sensitive to noise, the notebook uses
Savitzky-Golay smoothing before differentiating. This improves the phase-space
representation and helps isolate damping terms:

```text
v term:     -0.05324
x v term:    0.12095
x^2 v term: -0.23613
```

This is a key transition in the analysis: the model stops being only a
frequency estimate and starts describing how the oscillation loses energy.

### 6. Envelope-Based Parameter Fit

The Hilbert envelope is used to fit a nonlinear damping law. The notebook
reports:

```text
gamma = 0.35865
eta   = 0.07722
R0    = 4.6580
```

Here, `gamma` represents linear damping and `eta` represents nonlinear damping.
This gives a physically interpretable description of the decay curve, not just a
black-box regression.

### 7. ODE Model and Phase Calibration

The notebook then moves from discovered equations to direct simulation. The
oscillator is written as a first-order ODE system and integrated numerically.

The model used in the final methodology is:

```text
x' = v
v' = -omega_n^2 x - beta x^3 - (gamma + eta x^2) v
```

A fine calibration step estimates:

```text
phase phi      = -0.0484 rad
omega          = 117.6848 rad/s
omega^2        = 13849.72
```

This step aligns the simulated trajectory with the experimental phase, which is
essential because a small frequency mismatch creates a large visual error over
many cycles.

### 8. Global Optimization

The final model-updating step minimizes the mean squared error between the
experimental displacement and the ODE simulation. The notebook uses
Nelder-Mead-style optimization and physical penalties to keep parameters
consistent.

The final reported parameters are:

```text
x0     = 4.2833 mm
v0     = -6.8996 mm/s
gamma  = 0.3629
eta    = 0.0809
omega^2 = 13907.74
beta   = -6.91
```

The organized figure below shows the final global fit, combining the noisy
signal, SVD-smoothed signal, optimized numerical model and analytical envelope:

![Duffing global envelope fit](../figures/advanced/duffing_global_envelope_fit.png)

The interpretation is consistent with the SINDy investigation:

- the linear stiffness dominates the response;
- nonlinear damping is relevant to the envelope decay;
- the Duffing cubic stiffness term is small in the final global fit;
- using real initial velocity improves the time-domain match.

## Project Story

The two files show the project moving from plotting experiments to model-based
identification.

`testes.py` is a visualization scratch: it verifies how contour plots represent
a scalar field with strong gradients. It is useful as a plotting experiment, but
not as the main engineering result.

`definitivo.ipynb` is the central experimental notebook. It builds a complete
pipeline:

```text
raw CSVs
  -> event detection
  -> time and offset correction
  -> visual inspection
  -> SVD denoising
  -> SINDy equation discovery
  -> Duffing/nonlinearity investigation
  -> Hilbert envelope fitting
  -> ODE simulation
  -> global parameter optimization
```

The strongest technical conclusion is that the measured system behaves mainly
as a linear oscillator in stiffness, with damping behavior that benefits from a
nonlinear term. In portfolio language, the notebook demonstrates signal
processing, numerical linear algebra, sparse system identification and
physics-informed model calibration applied to a real home-built experiment.

## Suggested Public Description

```text
I built a vibration-analysis pipeline from a home experiment, starting with raw
CSV signals and ending with an interpretable oscillator model. The workflow
includes event detection, offset correction, SVD denoising, SINDy equation
discovery, Hilbert-envelope damping estimation and ODE-based global parameter
optimization. The final analysis indicates a predominantly linear stiffness
response with relevant nonlinear damping in the energy decay.
```
