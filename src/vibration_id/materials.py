from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FIRST_CANTILEVER_MODE_BETA = 1.87510407

# Rayleigh effective-mass fraction of a uniform cantilever for the first mode.
# Chosen so the lumped model k=3EI/L^3, m_eff=RAYLEIGH_BEAM_MASS_FRACTION*rho*A*L
# reproduces the continuous first-mode frequency when there is no tip mass.
RAYLEIGH_BEAM_MASS_FRACTION = 0.2427


@dataclass(frozen=True)
class BeamGeometry:
    """Rectangular cantilever beam geometry in SI units."""

    length_m: float
    thickness_m: float
    width_m: float = 0.025


@dataclass(frozen=True)
class MaterialCandidate:
    """Candidate material with an approximate Young modulus interval."""

    name: str
    young_min_gpa: float
    young_max_gpa: float
    density_kg_m3: float


@dataclass(frozen=True)
class MaterialYoungMatch:
    candidate: MaterialCandidate
    young_modulus_gpa: float
    distance_gpa: float
    within_range: bool


@dataclass(frozen=True)
class MaterialFrequencyMatch:
    candidate: MaterialCandidate
    observed_frequency_hz: float
    frequency_min_hz: float
    frequency_max_hz: float
    distance_hz: float
    within_range: bool


DEFAULT_MATERIALS = (
    MaterialCandidate("Polyethylene", 0.2, 0.7, 950.0),
    MaterialCandidate("Polypropylene", 1.3, 1.8, 900.0),
    MaterialCandidate("ABS", 1.8, 2.4, 1040.0),
    MaterialCandidate("PVC", 2.4, 4.0, 1400.0),
    MaterialCandidate("Acrylic (PMMA)", 2.7, 3.5, 1180.0),
    MaterialCandidate("Polystyrene", 3.0, 3.5, 1050.0),
    MaterialCandidate("Aluminum", 68.0, 72.0, 2700.0),
    MaterialCandidate("Stainless steel", 190.0, 210.0, 7900.0),
    MaterialCandidate("Carbon steel", 190.0, 210.0, 7850.0),
)


def _require_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def validate_geometry(geometry: BeamGeometry) -> None:
    _require_positive("length_m", geometry.length_m)
    _require_positive("thickness_m", geometry.thickness_m)
    _require_positive("width_m", geometry.width_m)


def section_area(geometry: BeamGeometry) -> float:
    """Return rectangular cross-section area in m^2."""

    validate_geometry(geometry)
    return geometry.width_m * geometry.thickness_m


def second_moment_area(geometry: BeamGeometry) -> float:
    """Return rectangular second moment of area in m^4."""

    validate_geometry(geometry)
    return geometry.width_m * geometry.thickness_m**3 / 12.0


def effective_tip_mass(
    geometry: BeamGeometry,
    *,
    density_kg_m3: float,
    tip_mass_kg: float = 0.0,
) -> float:
    """Rayleigh effective mass: ``0.2427 * rho * A * L + tip_mass``."""

    beam_mass = density_kg_m3 * section_area(geometry) * geometry.length_m
    return float(RAYLEIGH_BEAM_MASS_FRACTION * beam_mass + max(0.0, tip_mass_kg))


def natural_frequency_cantilever(
    geometry: BeamGeometry,
    *,
    young_modulus_pa: float,
    density_kg_m3: float,
    beta: float = FIRST_CANTILEVER_MODE_BETA,
    tip_mass_kg: float = 0.0,
) -> float:
    """Estimate first natural frequency with the Euler-Bernoulli beam model.

    With ``tip_mass_kg > 0`` a lumped Rayleigh model is used
    (``k = 3 E I / L^3``, ``m_eff = 0.2427 rho A L + m_tip``), which lowers the
    natural frequency to account for a mass attached at the free tip. With
    ``tip_mass_kg = 0`` it reduces exactly to the continuous first-mode formula.
    """

    validate_geometry(geometry)
    _require_positive("young_modulus_pa", young_modulus_pa)
    _require_positive("density_kg_m3", density_kg_m3)
    _require_positive("beta", beta)

    area = section_area(geometry)
    inertia = second_moment_area(geometry)
    if tip_mass_kg <= 0.0:
        rigidity_term = np.sqrt((young_modulus_pa * inertia) / (density_kg_m3 * area))
        return float((beta**2 / (2.0 * np.pi * geometry.length_m**2)) * rigidity_term)

    stiffness = 3.0 * young_modulus_pa * inertia / geometry.length_m**3
    m_eff = effective_tip_mass(geometry, density_kg_m3=density_kg_m3, tip_mass_kg=tip_mass_kg)
    return float(np.sqrt(stiffness / m_eff) / (2.0 * np.pi))


def young_modulus_from_frequency(
    geometry: BeamGeometry,
    *,
    frequency_hz: float,
    density_kg_m3: float,
    beta: float = FIRST_CANTILEVER_MODE_BETA,
    tip_mass_kg: float = 0.0,
) -> float:
    """Invert the Euler-Bernoulli first-mode formula to estimate Young modulus.

    Honors a tip mass via the same lumped Rayleigh model as
    :func:`natural_frequency_cantilever`. Ignoring a real tip mass biases the
    estimated modulus strongly low, since the attached mass lowers the measured
    frequency.
    """

    validate_geometry(geometry)
    _require_positive("frequency_hz", frequency_hz)
    _require_positive("density_kg_m3", density_kg_m3)
    _require_positive("beta", beta)

    if tip_mass_kg <= 0.0:
        numerator = 48.0 * np.pi**2 * density_kg_m3 * geometry.length_m**4 * frequency_hz**2
        denominator = geometry.thickness_m**2 * beta**4
        return float(numerator / denominator)

    inertia = second_moment_area(geometry)
    m_eff = effective_tip_mass(geometry, density_kg_m3=density_kg_m3, tip_mass_kg=tip_mass_kg)
    stiffness = (2.0 * np.pi * frequency_hz) ** 2 * m_eff
    return float(stiffness * geometry.length_m**3 / (3.0 * inertia))


def rank_materials_by_young(
    young_modulus_pa: float,
    candidates: tuple[MaterialCandidate, ...] = DEFAULT_MATERIALS,
) -> list[MaterialYoungMatch]:
    """Rank materials by proximity to an estimated Young modulus."""

    _require_positive("young_modulus_pa", young_modulus_pa)
    young_gpa = young_modulus_pa / 1e9
    matches: list[MaterialYoungMatch] = []
    for candidate in candidates:
        within = candidate.young_min_gpa <= young_gpa <= candidate.young_max_gpa
        if within:
            distance = 0.0
        else:
            distance = min(
                abs(young_gpa - candidate.young_min_gpa),
                abs(young_gpa - candidate.young_max_gpa),
            )
        matches.append(
            MaterialYoungMatch(
                candidate=candidate,
                young_modulus_gpa=young_gpa,
                distance_gpa=float(distance),
                within_range=within,
            )
        )
    return sorted(matches, key=lambda match: (match.distance_gpa, match.candidate.name))


def rank_materials_by_frequency(
    geometry: BeamGeometry,
    *,
    frequency_hz: float,
    candidates: tuple[MaterialCandidate, ...] = DEFAULT_MATERIALS,
    beta: float = FIRST_CANTILEVER_MODE_BETA,
    tip_mass_kg: float = 0.0,
) -> list[MaterialFrequencyMatch]:
    """Rank materials by the first-mode frequency interval predicted for each one."""

    validate_geometry(geometry)
    _require_positive("frequency_hz", frequency_hz)

    matches: list[MaterialFrequencyMatch] = []
    for candidate in candidates:
        f_min = natural_frequency_cantilever(
            geometry,
            young_modulus_pa=candidate.young_min_gpa * 1e9,
            density_kg_m3=candidate.density_kg_m3,
            beta=beta,
            tip_mass_kg=tip_mass_kg,
        )
        f_max = natural_frequency_cantilever(
            geometry,
            young_modulus_pa=candidate.young_max_gpa * 1e9,
            density_kg_m3=candidate.density_kg_m3,
            beta=beta,
            tip_mass_kg=tip_mass_kg,
        )
        low, high = sorted((f_min, f_max))
        within = low <= frequency_hz <= high
        distance = 0.0 if within else min(abs(frequency_hz - low), abs(frequency_hz - high))
        matches.append(
            MaterialFrequencyMatch(
                candidate=candidate,
                observed_frequency_hz=frequency_hz,
                frequency_min_hz=float(low),
                frequency_max_hz=float(high),
                distance_hz=float(distance),
                within_range=within,
            )
        )
    return sorted(matches, key=lambda match: (match.distance_hz, match.candidate.name))
