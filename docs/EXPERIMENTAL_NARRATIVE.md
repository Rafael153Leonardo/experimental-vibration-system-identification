# Technical Narrative: Experimental Acquisition, Sensor Calibration and System Identification

This project grew out of a homemade experimental setup to measure mechanical
vibrations and turn the acquired signal into interpretable physical parameters.
I built the bench, adapted the optical sensor, acquired the data with an Arduino,
and developed the computational pipeline in Python for filtering, system
identification and physical-model fitting.

The main goal was to start from a real raw signal and arrive at a quantitative
description of the system: natural frequency, damping, nonlinear sensor response
and an approximate dynamic model of the oscillator.

## 1. Experimental Acquisition Setup

The bench was assembled with a simple mechanical structure, a vibrating
blade/ruler and an optical target fixed at the tip of the system. The target's
displacement was measured laterally by a modified TCRT5000 reflective sensor
connected to an Arduino Uno.

![Full acquisition rig](../figures/setup/acquisition_rig_full.jpeg)

![Arduino and sensor circuit](../figures/setup/arduino_sensor_circuit.jpeg)

![Sensor and moving target alignment](../figures/setup/sensor_target_alignment.jpeg)

I used the Arduino only as an acquisition system. It did not apply any
calibration, filtering or conversion to displacement. The board read the raw
analog value from the sensor, recorded the time in microseconds and streamed the
data over serial as CSV. All signal interpretation was done later in Python.

That choice kept the embedded stage simple and deterministic. The Arduino was
responsible for sampling the sensor regularly, while the computer handled noise,
signal synchronization, frequency estimation and model fitting.

## 2. Arduino Acquisition Code

The code used on the Arduino is saved at `hardware/arduino/AMM.ino`. The routine
uses analog input `A0`, sets a sampling period of `1000 us` (roughly `1000 Hz`),
speeds up the ADC reading and streams two columns: `tempo_us` and
`leitura_bruta`.

```cpp
const int sensorPin = A0;
const unsigned long Ts_us = 1000; // 1000 Hz

unsigned long t0;
unsigned long nextSampleTime;

void setup() {
  Serial.begin(2000000);
  pinMode(sensorPin, INPUT);

  ADCSRA &= ~(bit(ADPS0) | bit(ADPS1) | bit(ADPS2));
  ADCSRA |= bit(ADPS2) | bit(ADPS0);

  t0 = micros();
  nextSampleTime = micros();

  Serial.println("tempo_us,leitura_bruta");
}

void loop() {
  unsigned long now = micros();

  if ((long)(now - nextSampleTime) >= 0) {
    int raw_value = analogRead(sensorPin);

    Serial.print(now - t0);
    Serial.print(",");
    Serial.println(raw_value);

    nextSampleTime += Ts_us;
  }
}
```

The main decisions at this stage were:

- use analog channel `A0` to capture the sensor's continuous response;
- work with `Ts_us = 1000`, keeping sampling around `1000 Hz`;
- use `Serial.begin(2000000)` to reduce the transmission bottleneck;
- change the ADC prescaler to cut `analogRead` latency;
- save the raw data to preserve the original experimental information.

## 3. Modifying and Calibrating the TCRT5000 Sensor

The TCRT5000 is a reflective optical sensor made of an infrared emitter and a
phototransistor. In common use it acts as a proximity or contrast detector. In
this project I used it differently: as an analog transducer to track the
displacement of a vibrating target.

To that end I modified the sensor's original use:

- positioned the optical pair relative to the moving target;
- mechanically adjusted distance and alignment;
- adapted the conditioning circuit for analog reading;
- used the Arduino to record the raw voltage;
- recomputed the effective response curve in Python from the real data.

The datasheet curve represents an idealized condition: relative phototransistor
current as a function of distance, using a standardized surface and test circuit.
The curve I obtained in the project is not the component's isolated curve. It
represents the effective response of the complete measurement chain:

```text
TCRT5000 + modified circuit + real target + mechanical alignment + Arduino ADC
```

This difference matters because the real experiment includes geometry, target
reflectivity, mounting, electrical noise and acquisition limitations.

## 4. Computational Reconstruction of the Sensor Response

After acquiring the vibration signal, I first fitted a linear model for the
dominant mechanical dynamics. That model describes the main part of the
oscillator: position, velocity, linear stiffness and linear damping.

Then I subtracted the identified linear behavior from the measured signal. The
remainder was not treated as mere noise. I analyzed that residual in time and in
phase space to separate the main mechanical dynamics from the nonlinearity
introduced by the measurement chain.

![SINDy linear model subtraction](../figures/sensor/sindy_linear_model_subtraction.png)

With this strategy, I used the linear-model residual to recover a nonlinear
correction associated with the sensor. The experimental curve `h(x)` represents
how the measured response deviates from the expected linear model.

![Sensor nonlinearity fit](../figures/sensor/sensor_nonlinearity_fit.png)

I also analyzed the residual's signature from different perspectives:

![Sensor residual signature](../figures/sensor/sensor_residual_signature.png)

![Sensor residual spectrum](../figures/sensor/sensor_residual_spectrum.png)

![Sensor residual phase space](../figures/sensor/sensor_residual_phase_space.png)

![Sensor residual projection](../figures/sensor/sensor_residual_projection.png)

The comparison with the TCRT5000 datasheet was done structurally and
qualitatively. The datasheet provides the component's standard optical curve; my
reconstruction provides the sensor's effective curve inside the real bench.

| TCRT5000 datasheet | Curve reconstructed in the experiment |
| --- | --- |
| Optical component in standard test condition | Complete measurement chain on the bench |
| Relative phototransistor current vs. distance | Residual correction `h(x)` vs. modeled state |
| Standardized reflective surface | Real target used in the vibrating system |
| Manufacturer-controlled geometry | Distance and alignment set by the assembly |
| Optical response of the component | Effective response: optics, circuit, mechanics and ADC |

## 5. Identification Pipeline

With the acquisition and sensor response characterized, I organized the pipeline
to turn the raw CSVs into physical results. The main flow was:

```text
raw data
  -> vibration onset detection
  -> time and offset correction
  -> visual inspection of the signals
  -> SVD filtering and reconstruction
  -> frequency estimation by FFT
  -> SINDy identification
  -> Hilbert envelope fit
  -> ODE simulation
  -> global parameter optimization
```

The `definitivo.ipynb` notebook concentrated the final stage of this analysis.
There, the synchronized signals were cleaned, compared and used to fit
progressively more structured dynamic models.

## 6. Data Preparation

At the start of the analysis, I loaded the synchronized CSVs and detected the
oscillation onset by the largest derivative jump. From that point, each signal
was shifted to start at `t = 0`.

I then applied basic corrections:

- zeroing the time axis;
- removing the signal offset;
- saving the corrected files;
- visually comparing experimental repetitions.

This step was important to ensure the identification was not done on misaligned
signals or with an artificial DC shift.

## 7. SVD Reconstruction

To reduce noise without destroying the dynamic structure, I used a
trajectory-matrix approach based on SVD. The time series was reorganized into
windows, decomposed into singular values and reconstructed with the most
relevant components.

In the main run I used:

```text
window L = 100
components kept = [0, 1]
```

The reconstruction produced the `x_svd_limpo` column, used as a cleaner input
for the identification stages.

## 8. SINDy Identification and Duffing Model

With the smoothed signal, I prepared the position and velocity states to apply
SINDy. The goal was to discover a candidate equation for the oscillator from the
data, without manually imposing all the terms from the start.

One of the identified models had a structure compatible with a Duffing-type
oscillator:

```text
(x)' = 1.000 v
(v)' = -20.285 -13892.495 x -0.508 v + 1.132 x^2 -0.214 x v + 4.082 x^3
```

I also compared different SINDy configurations:

```text
Model A R2 = 0.99973
Model B R2 = 0.99974
```

The physical interpretation was that the linear stiffness dominates the response,
while the nonlinear terms appear more relevantly in the energy dissipation and in
the sensor's residual correction.

## 9. Smoothed Derivatives and Nonlinear Damping

Since the velocity estimated by numerical differentiation is sensitive to noise,
I used Savitzky-Golay smoothing before computing derivatives. That improved the
phase-space portrait and allowed evaluating state-dependent damping terms.

The terms estimated at this stage were:

```text
v term:     -0.05324
x v term:    0.12095
x^2 v term: -0.23613
```

This part of the analysis showed that the system's energy loss did not need to
be described by linear damping alone.

## 10. Hilbert Envelope Fit

I used the Hilbert transform to extract the envelope of the damped signal and fit
a decay law. The fit returned:

```text
gamma = 0.35865
eta   = 0.07722
R0    = 4.6580
```

In this model, `gamma` represents the linear damping and `eta` represents a
nonlinear damping component. This turned the decay curve into interpretable
physical parameters.

## 11. ODE Simulation and Global Optimization

In the final stage, I wrote the oscillator as a first-order ODE system:

```text
x' = v
v' = -omega_n^2 x - beta x^3 - (gamma + eta x^2) v
```

I then integrated this model numerically and tuned the parameters to minimize the
error between the simulation and the experimental signal. I also refined phase and
frequency to align the simulated trajectory with the real data.

The fine calibration indicated:

```text
phase phi      = -0.0484 rad
omega          = 117.6848 rad/s
omega^2        = 13849.72
```

The global fit returned:

```text
x0      = 4.2833 mm
v0      = -6.8996 mm/s
gamma   = 0.3629
eta     = 0.0809
omega^2 = 13907.74
beta    = -6.91
```

The figure below summarizes this result, comparing the noisy signal, the
SVD-smoothed signal, the optimized numerical model and the analytic envelope:

![Duffing global envelope fit](../figures/advanced/duffing_global_envelope_fit.png)

## 12. Final Interpretation

The main result of the project was building a complete experimental
identification chain:

- assembled the acquisition bench;
- adapted the TCRT5000 for analog use;
- recorded real data with the Arduino;
- handled noise, offset and synchronization in Python;
- used the FFT to estimate the dominant frequency;
- used SVD to reconstruct cleaner signals;
- used SINDy to investigate the dynamic structure;
- used Hilbert to estimate damping;
- used ODE simulation to fit a global physical model.

The technical conclusion is that the measured system behaves mainly as a
linear-stiffness oscillator, but the energy dissipation and the measurement chain
present relevant nonlinear effects. The residual reconstruction made it possible
to separate part of the sensor nonlinearity from the main mechanical dynamics,
turning the sensor into a modeled element of the experiment rather than just a
source of raw data.

## 13. Original Sources Used

The original files used to organize this narrative were:

```text
C:\Users\rafael\Documents\arduino\AMM\AMM.ino
C:\Users\rafael\Downloads\TCRT5000.PDF
C:\Users\rafael\PycharmProjects\pythonProject3_TCC\prof\definitivo.ipynb
```

The TCRT5000 datasheet was used as a technical reference and is not redistributed
in this repository.
