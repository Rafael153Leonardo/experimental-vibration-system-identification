"""A physics-based digital twin of the clamped-ruler vibration rig.

The twin composes the three pieces the project identified separately into one
end-to-end generative model, from physical parameters to a synthetic *sensor*
reading:

    (E, rho, L, b, h, tip mass)                    physical parameters
        -> Euler-Bernoulli modal frequencies       f_n = f_1 * (beta_n/beta_1)^2
        -> nonlinear-damped oscillator in time      v' = -w^2 q - (gamma + eta q^2) v
        -> static cubic sensor map                  y = g (q + c q^3)
        -> synthetic "measured" signal

Every ingredient is grounded in a repo finding: the stiffness is **linear** (the
backbone is flat, so ``beta = 0``), the dissipation is **amplitude-dependent**
(``eta > 0``), the modal ratios follow the ideal clamped-free ladder, and the
measurement chain has a static cubic nonlinearity. Seeded with the identified
parameters the twin reproduces the real ruler; changing the parameters runs
virtual experiments (pluck, forced resonance sweep, geometry/material what-if,
tip-mass sensing).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vibration_id.beam_modes import cantilever_mode_ratios
from vibration_id.global_fit import duffing_rhs
from vibration_id.materials import (
    BeamGeometry,
    natural_frequency_cantilever,
    rank_materials_by_young,
    young_modulus_from_frequency,
)


@dataclass(frozen=True)
class BeamParams:
    """Physical parameters of the clamped ruler (SI units).

    Defaults are the identified stainless-steel ruler: ``E = 205.3`` GPa,
    ``L = 300`` mm, ``h = 0.55`` mm, ``b = 25`` mm, with a ~0.21 g paper tip
    target -- the combination that lands the fundamental at ~4.98 Hz.
    """

    young_modulus_pa: float = 205.3e9
    density_kg_m3: float = 7850.0
    length_m: float = 0.300
    thickness_m: float = 0.55e-3
    width_m: float = 0.025
    tip_mass_kg: float = 0.21e-3

    @property
    def geometry(self) -> BeamGeometry:
        return BeamGeometry(length_m=self.length_m, thickness_m=self.thickness_m, width_m=self.width_m)


@dataclass(frozen=True)
class DampingParams:
    """Oscillator damping ``v' = ... - (gamma + eta q^2) v``.

    ``gamma`` sets the low-amplitude decay (hence the quality factor
    ``Q = 2 pi f_1 / gamma``); ``eta`` adds the amplitude-dependent (drag-like)
    dissipation the ring-down envelope revealed. The defaults are calibrated to
    the inox sample (Q ~ 110 at a ~25 mm pluck, with the linear-viscous law
    losing to the amplitude-dependent one, R^2 ~ 0.95 vs ~1.0).
    """

    gamma: float = 0.16
    eta: float = 5.0e-3


@dataclass(frozen=True)
class SensorParams:
    """Static cubic sensor output map ``y = gain * (q + cubic * q^3)``.

    A pure odd cubic distortion of the true displacement -- the effective
    nonlinearity of the optics + circuit + target + ADC chain. ``cubic`` is in
    units of ``1 / q^2`` (per mm^2 when ``q`` is in mm).
    """

    gain: float = 1.0
    cubic: float = 3.0e-4

    def apply(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        return self.gain * (q + self.cubic * q**3)


@dataclass(frozen=True)
class FreeDecay:
    t: np.ndarray
    displacement: np.ndarray  # true tip displacement q(t)
    signal: np.ndarray  # sensor reading y(t) = h(q)


@dataclass(frozen=True)
class ForcedResponse:
    t: np.ndarray
    displacement: np.ndarray
    signal: np.ndarray
    drive_hz: float


@dataclass(frozen=True)
class DigitalTwin:
    """End-to-end simulator of the ruler rig from physical parameters."""

    beam: BeamParams = field(default_factory=BeamParams)
    damping: DampingParams = field(default_factory=DampingParams)
    sensor: SensorParams = field(default_factory=SensorParams)

    # -- modal structure -----------------------------------------------------
    def fundamental_hz(self, *, tip_mass_kg: float | None = None) -> float:
        """First natural frequency from the Euler-Bernoulli model (with tip mass)."""

        tip = self.beam.tip_mass_kg if tip_mass_kg is None else tip_mass_kg
        return natural_frequency_cantilever(
            self.beam.geometry,
            young_modulus_pa=self.beam.young_modulus_pa,
            density_kg_m3=self.beam.density_kg_m3,
            tip_mass_kg=tip,
        )

    def natural_frequencies(self, n_modes: int = 4) -> np.ndarray:
        """First ``n_modes`` frequencies on the ideal clamped-free ladder.

        The higher modes follow ``f_1 * (beta_n/beta_1)^2``; a small tip mass
        shifts mainly the fundamental (a ~3% effect), so this is the ideal-clamp
        ladder anchored on the tip-loaded fundamental.
        """

        return self.fundamental_hz() * cantilever_mode_ratios(n_modes)

    # -- virtual experiment: free-decay pluck --------------------------------
    def simulate_free_decay(
        self,
        *,
        amplitude_mm: float = 20.0,
        v0: float = 0.0,
        duration_s: float = 20.0,
        fs: float = 1000.0,
        apply_sensor: bool = True,
    ) -> FreeDecay:
        """Integrate a plucked ring-down: linear stiffness, nonlinear damping."""

        from scipy.integrate import odeint

        omega0_sq = (2.0 * np.pi * self.fundamental_hz()) ** 2
        t = np.arange(0.0, duration_s, 1.0 / fs)
        sol = odeint(
            duffing_rhs,
            [float(amplitude_mm), float(v0)],
            t,
            args=(omega0_sq, 0.0, self.damping.gamma, self.damping.eta),
        )
        q = sol[:, 0]
        y = self.sensor.apply(q) if apply_sensor else q
        return FreeDecay(t=t, displacement=q, signal=y)

    # -- virtual experiment: forced excitation / resonance sweep -------------
    def frequency_response(
        self,
        freqs_hz: np.ndarray,
        *,
        n_modes: int = 4,
        modal_weights: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Steady-state receptance magnitude and phase at each drive frequency.

        Sum of single-mode receptances with a constant modal damping ``gamma``
        (so ``Q_n = omega_n / gamma`` grows with mode number, matching the
        measured Q-vs-frequency trend). Returns ``(magnitude, phase_rad)``.
        """

        freqs_hz = np.asarray(freqs_hz, dtype=float)
        omega = 2.0 * np.pi * freqs_hz
        modes = self.natural_frequencies(n_modes)
        omega_n = 2.0 * np.pi * modes
        if modal_weights is None:
            modal_weights = np.ones(n_modes)
        modal_weights = np.asarray(modal_weights, dtype=float)

        h = np.zeros_like(omega, dtype=complex)
        for w_n, weight in zip(omega_n, modal_weights, strict=False):
            h += weight / (w_n**2 - omega**2 + 1j * self.damping.gamma * omega)
        return np.abs(h), np.angle(h)

    def simulate_forced(
        self,
        *,
        drive_hz: float,
        force: float = 5.0e3,
        duration_s: float = 10.0,
        fs: float = 1000.0,
        apply_sensor: bool = True,
    ) -> ForcedResponse:
        """Drive the mode nearest ``drive_hz`` and return the steady oscillation."""

        from scipy.integrate import odeint

        modes = self.natural_frequencies(max(4, 1))
        f_mode = float(modes[int(np.argmin(np.abs(modes - drive_hz)))])
        omega0_sq = (2.0 * np.pi * f_mode) ** 2
        drive = 2.0 * np.pi * drive_hz
        gamma, eta = self.damping.gamma, self.damping.eta

        def rhs(state: np.ndarray, tt: float) -> list[float]:
            q, v = state
            return [v, -omega0_sq * q - (gamma + eta * q**2) * v + force * np.cos(drive * tt)]

        t = np.arange(0.0, duration_s, 1.0 / fs)
        sol = odeint(rhs, [0.0, 0.0], t)
        q = sol[:, 0]
        y = self.sensor.apply(q) if apply_sensor else q
        return ForcedResponse(t=t, displacement=q, signal=y, drive_hz=float(drive_hz))

    # -- virtual experiment: tip-mass sensing --------------------------------
    def tip_mass_frequency_shift(self, added_mass_kg: np.ndarray) -> np.ndarray:
        """Fundamental frequency vs added tip mass (a mass-sensing calibration)."""

        added = np.atleast_1d(np.asarray(added_mass_kg, dtype=float))
        base = self.beam.tip_mass_kg
        return np.array([self.fundamental_hz(tip_mass_kg=base + dm) for dm in added])

    # -- virtual experiment: geometry / material what-if ---------------------
    def young_modulus_estimate(self) -> float:
        """Modulus the inverse EB model would return for this twin's frequency."""

        return young_modulus_from_frequency(
            self.beam.geometry,
            frequency_hz=self.fundamental_hz(),
            density_kg_m3=self.beam.density_kg_m3,
            tip_mass_kg=self.beam.tip_mass_kg,
        )

    def material_verdict(self) -> str:
        """Closest material to the twin's inferred modulus (uses each candidate's rho)."""

        matches = rank_materials_by_young(self.young_modulus_estimate())
        return matches[0].candidate.name
