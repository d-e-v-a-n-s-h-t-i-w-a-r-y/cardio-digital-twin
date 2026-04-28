"""Probabilistic input generators for cardiovascular mission simulation.

All stochastic helpers accept an explicit ``np.random.Generator`` so the
simulation is fully reproducible from a seed.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Basic variability helpers
# ---------------------------------------------------------------------------

def normal_variation(
    base: float,
    sd: float,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw values from a normal distribution around a baseline."""
    return rng.normal(loc=base, scale=sd, size=size)


def poisson_stress_events(
    rate_per_hour: float,
    duration_hours: float,
    rng: np.random.Generator,
) -> int:
    """Number of random stress events over *duration_hours* via Poisson model."""
    lam = max(0.0, rate_per_hour * duration_hours)
    return int(rng.poisson(lam=lam))


def lognormal_exercise_intensity(
    mu: float,
    sigma: float,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample exercise intensity (0–∞, typically 0–2) from a log-normal distribution.

    mu and sigma are the *underlying normal* parameters (not the LN mean/std).
    """
    return np.clip(rng.lognormal(mean=mu, sigma=sigma, size=size), 0.0, 3.0)


def bounded(
    value: np.ndarray | float,
    lower: float | None = None,
    upper: float | None = None,
) -> np.ndarray | float:
    """Clip a value or array into a physiologic range."""
    return np.clip(value, a_min=lower, a_max=upper)


# ---------------------------------------------------------------------------
# G-force profile helpers
# ---------------------------------------------------------------------------

# Soyuz / GAGANYAAN nominal re-entry profile (empirically derived from
# published flight data, IndJAerospaceMed, Kglmeridian.com, USRA reports):
#   - Peak +Gx ≈ 4.2 G sustained for ~90 s during peak heating
#   - Ballistic abort trajectory: up to 8.2 G
#   - Launch (Soyuz / LVM3): 0 → 3.2 G over ~520 s ascent

REENTRY_PEAK_GX = 4.2        # nominal chest-to-back (+Gx), G-units
LAUNCH_PEAK_GZ = 3.2         # head-to-foot (+Gz during ascent, G-units)

# Gz baroreceptor model (Stoll 1956, refined by Whinnery 1987):
#   HR increases linearly with +Gz as the baroreceptor reflex fires to
#   compensate for reduced cerebral perfusion pressure.
#   Empirically: ΔHR ≈ +8 bpm per 1 G (above 1 G) for unprotected subjects.
#   MAP at heart level ≈ MAP₀ + ρ·g·h_hs·(Gz−1)
#   where h_hs = vertical distance heart→carotid sinus ≈ 0.25 m,
#   ρ_blood ≈ 1060 kg/m³, g = 9.81 m/s².
G = 9.81                     # m/s²
RHO_BLOOD = 1060.0           # kg/m³ — density of blood
H_HEART_TO_SINUS = 0.25      # m    — ~25 cm heart-to-carotid sinus
H_HEART_TO_FEET = 1.20       # m    — ~120 cm heart-to-ankle (venous pooling)
MMHG_PER_PA = 1.0 / 133.322  # 1 mmHg = 133.322 Pa

# Hydrostatic pressure gradient at heart from footward pools under +Gz
#   ΔP_pool = ρ·g·h_feet·(Gz−1)    [Pa], positive = raises venous pooling
# Corresponding MAP drop at brain relies on:
#   ΔP_brain = ρ·g·h_sinus·(Gz−1) [Pa]  — loss of perfusion head pressure

def gz_cardiovascular_delta(
    gz: float,
    *,
    hr_base: float = 70.0,
    map_base: float = 93.0,
    co_base: float = 5.0,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Compute ΔHR, ΔMAP, ΔCO caused by the given +Gz load (head-to-foot G).

    Physics basis
    -------------
    The hydrostatic pressure drop at the carotid (brain) level:
        ΔP_cerebral = ρ · g · h_hs · (Gz − 1)        [Pa]
    MAP must *increase* by this amount to maintain cerebral perfusion, so:
        ΔMAP_required = ΔP_cerebral · mmHg_per_Pa      [mmHg]
    Baroreceptor tachycardia to compensate:
        ΔHR ≈ k_baroref · (Gz − 1)                    [bpm]
        k_baroref ≈ 8 bpm/G  (Whinnery 1987)
    Cardiac output (Frank-Starling + peripheral resistance trade-off):
        ΔCO = −(venous_pooling_fraction) · co_base
        venous_pooling_fraction ≈ 0.05 · (Gz − 1) per G  (up to ~30 % at 7G)

    Parameters
    ----------
    gz : float
        Effective +Gz load (1.0 = normal Earth gravity, 0.0 = microgravity).
    hr_base, map_base, co_base : float
        Baseline physiologic values for contextual non-linear scaling.
    rng : optional Generator
        If provided, adds ±5 % physiologic noise to each delta.

    Returns
    -------
    (delta_hr, delta_map, delta_co) – signed mmHg / bpm / L·min⁻¹
    """
    gz_excess = gz - 1.0  # force above 1 G; negative in microgravity

    # --- Baroreceptor-driven tachycardia / bradycardia ---
    # Above 1 G: compensatory tachycardia (+8 bpm/G, capped at ~40 bpm max).
    # Below 1 G: mild bradycardia (−3 bpm at 0 G due to cephalad shift + reduced venous return).
    if gz_excess >= 0:
        k_baroref = 8.0  # bpm per G (Whinnery 1987 +Gz data)
        delta_hr = k_baroref * gz_excess
    else:
        # microgravity: moderate bradycardia, net effect dominates toward slight HR rise
        # from fluid redistribution (NASA ISS studies show ~small net increase
        # in early mission vs significant acute resting HR reduction in bedrest).
        # Model: mild baroreceptor bradycardia partially offset by sympathetic activation.
        delta_hr = 2.5 * gz_excess  # e.g. −2.5 bpm at 0 G

    # --- Hydrostatic MAP effect at heart level ---
    # +Gz → blood pools footward → MAP at heart may marginally rise,
    # but cerebral MAP drops (brain is above heart):
    #   ΔP_cerebral [Pa] = ρ·g·h_hs·gz_excess
    dp_cerebral_pa = RHO_BLOOD * G * H_HEART_TO_SINUS * gz_excess
    delta_map = dp_cerebral_pa * MMHG_PER_PA  # positive = more work needed

    # --- Cardiac output: venous pooling reduces preload ---
    # Fraction of blood pooled in legs ≈ 0.05 per G (up to 0.30 at 7 G)
    # CO drops proportionally to reduced preload via Frank-Starling
    pool_frac = np.clip(0.06 * gz_excess, -0.15, 0.30)
    delta_co = -pool_frac * co_base

    # Optional noise
    if rng is not None:
        noise_scale = 0.05
        delta_hr *= 1.0 + rng.uniform(-noise_scale, noise_scale)
        delta_map *= 1.0 + rng.uniform(-noise_scale, noise_scale)
        delta_co *= 1.0 + rng.uniform(-noise_scale, noise_scale)

    return float(delta_hr), float(delta_map), float(delta_co)


def gx_cardiovascular_delta(
    gx: float,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Compute ΔHR, ΔMAP, ΔCO for +Gx (chest-to-back) acceleration.

    +Gx (eyeballs-in) is experienced during Soyuz/GAGANYAAN re-entry.
    It is much better tolerated than +Gz because blood does not pool
    away from the brain (horizontal orientation).

    Physics basis
    -------------
    Blood pools along the antero-posterior axis (~0.20 m typical depth).
    Cardiac work increases because the heart must maintain output against
    chest compression; HR rises ≈ 5 bpm/G above 1 G.
    MAP rises modestly (thoracic compression raises systemic resistance).

    Parameters
    ----------
    gx : float
        Chest-to-back G-load (1.0 = upright standing, 4.2 = peak re-entry).

    Returns
    -------
    (delta_hr, delta_map, delta_co)
    """
    gx_excess = max(0.0, gx - 1.0)

    # HR rise under +Gx (less severe than +Gz; ≈5 bpm/G)
    delta_hr = 5.0 * gx_excess

    # MAP slightly rises due to chest compression and increased sympathetic tone
    # ρ·g·h_AP·gx_excess where h_AP ≈ 0.20 m antero-posterior distance
    H_AP = 0.20
    dp_pa = RHO_BLOOD * G * H_AP * gx_excess
    delta_map = dp_pa * MMHG_PER_PA

    # CO mildly reduced by reduced venous return under thoracic loading
    delta_co = -0.03 * gx_excess * 5.0  # ≈ −0.15 L/min at 4.2 G peak re-entry

    if rng is not None:
        noise_scale = 0.07
        delta_hr *= 1.0 + rng.uniform(-noise_scale, noise_scale)
        delta_map *= 1.0 + rng.uniform(-noise_scale, noise_scale)
        delta_co *= 1.0 + rng.uniform(-noise_scale, noise_scale)

    return float(delta_hr), float(delta_map), float(delta_co)
