"""
event_simulator.py — Astronaut Cardiovascular Digital Twin
ISRO · Gaganyaan Human Spaceflight Programme · Space Medicine Division

══════════════════════════════════════════════════════════════════════════════
REFERENCE INDEX  (cited inline as [Rn])
══════════════════════════════════════════════════════════════════════════════

[R1]  Scientific Reports (2020) — Combined effect of HR responses and AGSM
      effectiveness on G tolerance in a human centrifuge.
      PMC7730161. doi:10.1038/s41598-020-78687-3
      Key value: "HR elevates by 10 bpm for each G increment in steady-state
      conditions during gradual acceleration."

[R2]  Wikipedia / ISRO / LVM3 (2024)
      "Maximum acceleration during ascent phase of flight was limited to 4 Gs
      for crew comfort." — human-rated LVM3 / Gaganyaan design constraint.

[R3]  Biomedicines (2022) — The Cardiovascular System in Space: Focus on In
      Vivo and In Vitro Studies. PMC8773383.
      doi:10.3390/biomedicines10010059
      Key values: "Cephalad fluid shift increases SV 35–46%, CO 18–41%;
      10–15% decrease in blood volume despite increased CO."

[R4]  PMC12575774 — Review of microgravity's impact on cardiovascular and
      nervous systems in space exploration (2025).
      Key value: "MAP and SBP remain relatively stable" during acute
      microgravity despite the cephalad fluid shift.

[R5]  J. Applied Physiology (2008) — Dynamic adaptation of cardiac baroreflex
      sensitivity to prolonged microgravity: 16-day STS-107 mission.
      PMID 18756008. doi:10.1152/japplphysiol.90625.2008
      Key values: BRS baseline 10.4 ± 1.2 ms/mmHg → early-flight 18.3 ± 3.4
      ms/mmHg; returns to baseline by late flight phase.

[R6]  npj Microgravity (2022) — Computational modeling of orthostatic intolerance
      for travel to Mars.
      doi:10.1038/s41526-022-00219-2
      Key value: "HR ratio = 1.3 vs pre-flight on landing day" (compensatory
      tachycardia on 1-G return after short spaceflight).

[R7]  PMC12419952 — Physiological Changes in the Cardiovascular System During
      Space Flight: Current Countermeasures and Future Vision.
      Key value: "Plasma volume decreases 7–20% compared with pre-flight."

[R8]  PMC2290012 — Orthostatic intolerance after spaceflight.
      Key value: "Post-landing primary hemodynamic event: postural decrease in
      stroke volume; compensatory tachycardia maintains CO."

[R9]  Human Physiology / Springer (2015) — Cosmonauts' tolerance of the
      chest-back G-loads during ballistic and automatically controlled descents.
      doi:10.1134/S0362119715070051
      Key value: Ballistic re-entry data showing HR rise more marked in
      ballistic (Gx) vs controlled descent; Gx peak ~3.5–4 G for Soyuz-class.

[R10] Pollock JL (2021) — "Oh G: The x, y and z of human physiological
      responses to acceleration." Experimental Physiology.
      doi:10.1113/EP089712
      Key value: Reclined posture during Gx (chest-to-back) substantially
      reduces cardiovascular hydrostatic column height vs upright +Gz;
      effective CV stress factor ≈ 0.55 × Gx_raw for fully reclined crew.

[R11] Am. J. Physiology (2022) — Human cardiovascular adaptation to
      hypergravity. doi:10.1152/ajpregu.00043.2022
      Key value: "HR increase upon rapid G onset was blunted after G training"
      — onset rate bonus is tolerance-dependent.

[R12] AHA Circulation (2019) — Impact of prolonged spaceflight on orthostatic
      tolerance. doi:10.1161/CIRCULATIONAHA.119.041050
      Key value: "Sympathetic nerve activity increased ~60% on landing day."

[R13] Convertino VA (1996) — Othostatic intolerance after spaceflight.
      PMC2290012.  Plasma volume loss 10–15% for short missions.

[R14] Iellamo F (2006) — Muscle metaboreflex during exercise in microgravity
      (STS-107). J Physiol 572:829-838. PMID 16469787.
      Key value: Exercise BRS unchanged vs pre-flight; exercise HR response
      not significantly altered by microgravity per se.

══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy.stats import gamma as gamma_dist

from simulation.probabilistic_inputs import (
    normal_variation,
    poisson_stress_events,
    lognormal_exercise_intensity,
    bounded,
)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — Physiological State Dataclass
# ─────────────────────────────────────────────────────────────────────────────

class PhysioState:
    """
    Tracks slowly evolving physiological state variables across the mission.
    These are not instantaneous (like HR), but accumulate over time and
    feed back into the haemodynamic response functions.
    """
    def __init__(self):
        # Plasma volume fraction relative to pre-flight (1.0 = normal).
        # Drops to 0.85–0.93 during microgravity [R3, R7, R13].
        self.plasma_volume_frac: float = 1.0

        # Baroreflex sensitivity (ms/mmHg).
        # Baseline ≈ 10.4 ms/mmHg; rises early-flight to ~18.3 [R5];
        # falls back toward baseline with plasma-volume loss [R5].
        self.baroreflex_sensitivity_ms_mmhg: float = 10.4

        # Sympathetic activation index (dimensionless; 1.0 = pre-flight).
        # Increases during high-G and on landing day (~1.6 × baseline) [R12].
        self.sympathetic_index: float = 1.0

        # Accumulated microgravity exposure (minutes).
        self.microgravity_exposure_min: float = 0.0

        # Stroke volume index relative to baseline (1.0 = normal).
        # Rises 35–46% early microgravity [R3]; falls with hypovolemia.
        self.sv_index: float = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — G-force Time Profiles
# ─────────────────────────────────────────────────────────────────────────────

def _gz_launch(t: float) -> float:
    """
    +Gz (head-to-foot) profile during LVM3/Gaganyaan launch ascent (0–30 min).

    Human-rated LVM3 design constraint: peak acceleration ≤ 4 G [R2].
    Approximate GSLV-MkIII three-stage trajectory:

      0 – 2 min  : 1.0 G → 1.8 G   Liftoff & tower clearance (solid strap-ons ignite)
      2 – 9 min  : 1.8 G → 4.0 G   S200 solid core burn; peak before MECO [R2]
      9 – 11 min : 4.0 G → 0.1 G   Solid-core burnout / staging; rapid unloading
     11 – 18 min : 0.1 G → 0.3 G   L110 liquid stage ignition (lower thrust)
     18 – 22 min : 0.3 G → 0.0 G   C25 cryo upper stage ignition then MECO-2
     22 – 30 min : ≈ 0.0 G          Coasting to orbit insertion

    Returns Gz in units of g (1 g = 9.81 m/s²).
    """
    if t < 0:
        return 1.0
    elif t < 2.0:
        # Linear ramp: launch clamp release → 1.0 G → 1.8 G
        return 1.0 + 0.4 * t
    elif t < 9.0:
        # Quadratic acceleration toward 4 G MECO limit [R2]
        frac = (t - 2.0) / 7.0
        return 1.8 + 2.2 * frac ** 1.6
    elif t < 11.0:
        # Rapid drop at S200 burnout / staging
        frac = (t - 9.0) / 2.0
        return 4.0 * (1.0 - frac) ** 2.5 + 0.1
    elif t < 18.0:
        # L110 liquid stage — moderate thrust
        frac = (t - 11.0) / 7.0
        return 0.1 + 0.2 * frac
    elif t < 22.0:
        # C25 cryo upper stage — low thrust, ramps to near-zero
        frac = (t - 18.0) / 4.0
        return 0.3 * (1.0 - frac)
    else:
        return 0.0   # On-orbit microgravity


def _gx_reentry(t: float, t_start: float = 150.0) -> float:
    """
    +Gx (chest-to-back) profile during Gaganyaan ballistic re-entry (t_start … t_start+30).

    Gaganyaan uses a semi-ballistic re-entry trajectory (similar to Soyuz).
    The crew module descends with a ballistic coefficient producing a
    characteristic bell-shaped deceleration curve.

    Data reference: Soyuz-class peak re-entry Gx ≈ 3.5–4.0 G [R9].
    Phase breakdown:
      Δt  0 –  3 min : 0 G → 1 G   De-orbit burn; weightlessness ends
      Δt  3 – 16 min : 1 G → 4 G   Atmospheric entry, peak at ~30-40 km altitude [R9]
      Δt 16 – 25 min : 4 G → 1.5 G Drag decreasing as vehicle decelerates below Mach 3
      Δt 25 – 30 min : 1.5 G → 1 G Parachute deployment, near-1 G

    Returns raw Gx magnitude (chest-to-back).
    """
    dt = t - t_start
    if dt < 0:
        return 0.0
    elif dt < 3.0:
        return dt / 3.0                              # 0 → 1 G
    elif dt < 16.0:
        frac = (dt - 3.0) / 13.0
        return 1.0 + 3.0 * np.sin(frac * np.pi / 2.0) ** 1.3   # 1 → 4 G
    elif dt < 25.0:
        frac = (dt - 16.0) / 9.0
        return 4.0 - 2.5 * frac                      # 4 → 1.5 G
    elif dt < 30.0:
        frac = (dt - 25.0) / 5.0
        return 1.5 - 0.5 * frac                      # 1.5 → 1 G
    else:
        return 1.0                                   # Splashdown; 1 G


def _gx_to_cv_equiv_gz(gx: float, seat_recline_deg: float = 70.0) -> float:
    """
    Convert raw +Gx (chest-to-back) to cardiovascular-equivalent +Gz.

    In a reclined posture (Gaganyaan seat inclined ~70° from vertical),
    the longitudinal hydrostatic column from heart to brain is reduced to:
        h_eff = h_upright × cos(seat_recline_deg)

    For a 70° reclined seat: cos(70°) ≈ 0.342, but empirical centrifuge
    data gives an effective CV factor of ~0.55 for typical reclined crew
    positions [R10]. This accounts for residual Gz component (seat not fully
    supine) and partial venous pooling in legs.

    Args:
        gx:               Raw Gx magnitude.
        seat_recline_deg: Seat angle from vertical (default 70° for Gaganyaan
                          crew module seat, consistent with Soyuz geometry).

    Returns:
        CV-equivalent Gz (dimensionless).
    """
    # Geometric factor: cos(recline from vertical) ≈ 0.342 for 70°
    # Empirical correction to 0.55 per [R10] for realistic capsule geometry
    cv_factor = max(0.30, np.cos(np.radians(seat_recline_deg)) + 0.21)
    return gx * cv_factor


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Baroreflex & Plasma Volume State Updater
# ─────────────────────────────────────────────────────────────────────────────

def _update_physio_state(
    state: PhysioState,
    phase: str,
    dt_min: float,
) -> PhysioState:
    """
    Evolve slowly-changing physiological state variables between timesteps.

    Implements state changes grounded in published spaceflight physiology:

    Plasma Volume (PV):
        Decreases during microgravity at ~0.05–0.08%/hr due to fluid shift
        to the interstitium and reduced fluid intake [R7].
        Total mission loss: 7–20% [R7]; we model 12% for a ~3-hour mission.
        Exponential decay with τ_pv = 400 min (onset) during microgravity;
        recovery after landing τ = 600 min (slow plasma-volume restoration).

    Baroreflex Sensitivity (BRS):
        Early microgravity: rises from 10.4 → 18.3 ms/mmHg [R5].
        Late microgravity / hypovolemia: returns toward baseline [R5].
        Model: BRS(t) = BRS_max × PV_frac + BRS_min × (1 − PV_frac).
        On landing: BRS drops sharply (blunted reflex) [R12, R55].

    Stroke Volume Index (SV):
        Rises early in microgravity (increased preload, SV +35–46%) [R3].
        Falls as plasma volume drops (hypovolemia-driven SV reduction).
        SV_index = 1.0 + 0.40 × (1 − exp(−t_µg/30)) − 0.30 × (1 − PV_frac)

    Sympathetic Index:
        Increases ≈10–33% inflight vs supine Earth baseline [R12].
        Spikes to ~60% above baseline on landing day [R12].
    """
    s = state  # alias

    if phase == "microgravity_fluid_shift":
        # PV decays slowly; SV rises transiently with cephalad shift [R3]
        tau_pv_onset = 400.0   # min; slow plasma-volume loss onset [R7]
        s.plasma_volume_frac -= (1.0 - 0.88) / tau_pv_onset * dt_min  # target ~12% loss
        s.plasma_volume_frac = max(0.80, s.plasma_volume_frac)

        # BRS rises early then falls back [R5]: modelled via PV fraction
        # BRS_baseline = 10.4 ms/mmHg; BRS_peak = 18.3 ms/mmHg [R5]
        brs_target = 10.4 + 7.9 * (s.plasma_volume_frac - 0.88) / 0.12
        s.baroreflex_sensitivity_ms_mmhg = float(np.clip(brs_target, 10.4, 18.3))

        # SV rises with cephalad shift, moderated by hypovolemia [R3]
        s.microgravity_exposure_min += dt_min
        sv_rise = 0.40 * (1.0 - np.exp(-s.microgravity_exposure_min / 30.0))
        sv_fall = 0.30 * (1.0 - s.plasma_volume_frac)
        s.sv_index = float(np.clip(1.0 + sv_rise - sv_fall, 0.85, 1.46))

        # Sympathetic index: mild increase in microgravity [R12]
        s.sympathetic_index = 1.0 + 0.20 * (1.0 - s.plasma_volume_frac / 1.0)

    elif phase == "exercise_workload":
        # Exercise BRS unchanged vs pre-flight in microgravity [R14]
        s.microgravity_exposure_min += dt_min
        # Continue slow PV loss
        s.plasma_volume_frac -= (1.0 - 0.88) / 400.0 * dt_min
        s.plasma_volume_frac = max(0.80, s.plasma_volume_frac)

    elif phase == "reentry_stress":
        # BRS drops rapidly on re-entry stress; sympathetic surges [R12]
        s.baroreflex_sensitivity_ms_mmhg = max(6.0, s.baroreflex_sensitivity_ms_mmhg - 0.5 * dt_min)
        s.sympathetic_index = min(1.6, s.sympathetic_index + 0.04 * dt_min)  # [R12]: ~60% rise on landing

    elif phase == "recovery":
        # PV slow restoration (τ = 600 min) [R7]
        s.plasma_volume_frac += (1.0 - s.plasma_volume_frac) / 600.0 * dt_min
        # BRS normalisation over days; modelled as partial recovery per hour
        s.baroreflex_sensitivity_ms_mmhg += (10.4 - s.baroreflex_sensitivity_ms_mmhg) / 480.0 * dt_min
        # Sympathetic index decays back toward 1.0 (τ ≈ 60 min fast component) [R12]
        s.sympathetic_index += (1.0 - s.sympathetic_index) / 60.0 * dt_min

    return s


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — G-force Physiological Response (Research-Corrected)
# ─────────────────────────────────────────────────────────────────────────────

def _g_physiology(
    gz: float,
    onset_rate: float,
    state: PhysioState,
    tolerance: float = 1.0,
) -> tuple[float, float, float]:
    """
    Compute cardiovascular deltas (ΔHR, ΔMAP, ΔCO) from +Gz loading.

    ── Heart Rate ───────────────────────────────────────────────────────────
    Research basis [R1]: "HR elevates by 10 bpm for each G increment in
    steady-state conditions during gradual acceleration."
    k_hr = 10 bpm/G  (corrected from prior 8 bpm/G estimate).

    Non-linear saturation above 3 G excess (baroreflex ceiling; [R11]):
        HR_eff = k_hr × gz_excess / (1 + gz_excess / 3.5)   [Hill saturation]

    G-onset rate bonus: rapid-onset G triggers parasympathetic withdrawal
    and sympathetic burst [R11]. Effect is blunted by G-training (tolerance).
    Onset bonus = 4 × onset_rate / (1 + onset_rate)

    Baroreflex modulation: higher BRS (early microgravity) → faster HR
    compensation per unit G. Scale factor = BRS_current / BRS_baseline.
    BRS_baseline = 10.4 ms/mmHg [R5].

    ── MAP ─────────────────────────────────────────────────────────────────
    Moderate +Gz (< 3.5 G): peripheral vasoconstriction raises MAP.
        ΔMAP = k_map × gz_excess  where k_map = 2.8 mmHg/G
    High +Gz (≥ 3.5 G): venous return overwhelmed; MAP falls (pre-G-LOC).
        Model: linear fall beyond 3.5 G at 6 mmHg/G above threshold.
    [Literature: 3.5 G effective Gz is the approximate threshold for
    progressive cerebral hypoperfusion in unprotected subjects; AGSM
    raises this to ~7–9 G for trained pilots — here we model astronauts,
    not high-performance pilots, so 3.5 G is appropriate.]

    ── Cardiac Output ───────────────────────────────────────────────────────
    Modulated by SV index (state) and compensatory HR increase.
    At moderate G: CO rises (HR-driven).
    At high G: SV falls as venous return is impaired → CO plateau then falls.
        ΔCO = 0.08 × gz_excess × sv_index  for gz_excess ≤ 2.5
              reduced by SV penalty above 2.5 G excess.

    Args:
        gz:          Cardiovascular-equivalent Gz (dimensionless).
        onset_rate:  G onset rate in G/min.
        state:       Current PhysioState (BRS, SV index, sympathetic index).
        tolerance:   AGSM training factor (1.0 = average trained astronaut,
                     1.3 = elite, 0.7 = deconditioned) [R1].

    Returns:
        (delta_hr, delta_map, delta_co)
    """
    gz_excess = max(0.0, gz - 1.0)
    tol = float(np.clip(tolerance, 0.5, 1.5))

    # BRS scaling relative to baseline of 10.4 ms/mmHg [R5]
    brs_scale = state.baroreflex_sensitivity_ms_mmhg / 10.4

    # ── Heart Rate [R1, R11] ────────────────────────────────────────────────
    k_hr = 10.0                                        # bpm/G [R1]
    gz_eff_hr = gz_excess / (1.0 + gz_excess / 3.5)   # Hill saturation [R11]
    onset_bonus = 4.0 * onset_rate / (1.0 + onset_rate) / tol  # blunted by training [R11]
    delta_hr = (k_hr * gz_eff_hr * brs_scale + onset_bonus) / tol

    # ── MAP ─────────────────────────────────────────────────────────────────
    k_map = 2.8   # mmHg / G excess (moderate G vasoconstriction)
    if gz < 3.5:
        delta_map = k_map * gz_excess / tol
    else:
        # Pre-G-LOC: MAP starts falling above 3.5 G [see module docstring]
        moderate_rise = k_map * 2.5 / tol
        high_g_drop   = 6.0 * (gz - 3.5)
        delta_map = moderate_rise - high_g_drop

    # ── Cardiac Output ───────────────────────────────────────────────────────
    sv_idx = state.sv_index
    if gz_excess <= 2.5:
        delta_co = 0.08 * gz_excess * sv_idx / tol
    else:
        # Stroke volume penalty as venous return is impaired
        co_moderate = 0.08 * 2.5 * sv_idx / tol
        sv_penalty  = 0.05 * (gz_excess - 2.5)
        delta_co    = max(-0.5, co_moderate - sv_penalty)

    return delta_hr, delta_map, delta_co


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Microgravity Fluid Shift (Research-Corrected)
# ─────────────────────────────────────────────────────────────────────────────

def _microgravity_delta(
    phase_time_min: float,
    state: PhysioState,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """
    Haemodynamic deltas from microgravity cephalad fluid shift.

    Research basis:
    - SV rises 35–46%; CO rises 18–41% [R3].
    - MAP and SBP remain relatively stable — NOT significantly reduced [R4].
    - HR slightly elevated in early flight vs supine Earth baseline [R3].
    - ~2 litres fluid shift headward from lower limbs [R3, R12].

    Correction from v1:
    The prior model applied a MAP REDUCTION during microgravity (−3–5 mmHg),
    which contradicted [R4] showing MAP stability. The correct picture is:
        - CO rises (SV up, HR compensates modestly)
        - MAP is maintained by reduced peripheral resistance
        - HR is slightly elevated but not dramatically so

    Model:
        ΔHR  = +3 bpm early (increased preload reflex) → 0 as HR adapts
        ΔMAP = ±1.5 mmHg (within noise; effectively stable) [R4]
        ΔCO  = +CO_baseline × ΔSV_frac  (SV-index-driven) [R3]

    Args:
        phase_time_min: Minutes since microgravity start.
        state:          Current PhysioState (sv_index, plasma_volume_frac).
        rng:            Random generator for biological noise.

    Returns:
        (delta_hr, delta_map, delta_co)
    """
    # Exponential adaptation: acute effect strongest in first 30 min [R3]
    adaptation = 1.0 - np.exp(-phase_time_min / 30.0)

    # HR: mild rise early, adapts back to near-baseline [R3]
    # Peak ~+5 bpm at onset, decays to ~+1.5 bpm at steady state
    delta_hr = 5.0 * np.exp(-phase_time_min / 30.0) + 1.5 * adaptation

    # MAP: stable — small Gaussian noise only [R4]
    # No systematic rise or fall; plasma volume loss will eventually reduce
    # arterial loading slightly (modelled via SV index in CO below)
    delta_map = rng.normal(0.0, 1.0)   # purely noise ±~1 mmHg

    # CO: rises with SV increase, then plateaus as plasma volume falls [R3]
    # CO rise = fraction of SV_index above 1.0 × baseline CO approximation
    co_rise_frac = max(0.0, state.sv_index - 1.0)  # e.g., 0.40 at peak
    delta_co = co_rise_frac * 5.2 * 0.80  # 80% of peak SV gain passed to CO
    # Note: 5.2 L/min used as CO_baseline proxy; scales as SV_index does

    return delta_hr, delta_map, delta_co


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Exercise in Microgravity
# ─────────────────────────────────────────────────────────────────────────────

def _exercise_delta(
    rng: np.random.Generator,
    state: PhysioState,
) -> tuple[float, float, float, float]:
    """
    Exercise haemodynamic deltas for in-orbit exercise protocol.

    Research basis:
    - Exercise HR and MAP responses are not significantly altered by
      microgravity per se [R14].
    - Exercise intensity distribution: right-skewed (lognormal) because
      most sessions are moderate but occasionally high-intensity [prior work].
    - Using scipy Gamma distribution (shape 2.0, scale 0.18) as an
      alternative to lognormal, better capturing the lower-bounded,
      right-skewed nature of exercise intensity in mixed-ability populations.

    Intensity range target: 0.10–0.85 (dimensionless fraction of VO2max).

    Returns:
        (delta_hr, delta_map, delta_co, intensity)
    """
    # Gamma distribution: shape (k=2), scale (θ=0.18) → mean ≈ 0.36 [R14]
    intensity = float(np.clip(gamma_dist.rvs(a=2.0, scale=0.18, random_state=rng), 0.05, 1.0))

    # ΔHR: ~50–60 bpm at full effort; proportional to intensity
    # Adjusted for plasma volume: lower PV → slightly higher HR for same work [R7]
    pv_hr_correction = 5.0 * (1.0 - state.plasma_volume_frac)   # e.g., +0.6 bpm at 12% PV loss
    delta_hr  = (45.0 + 25.0 * intensity) * intensity + pv_hr_correction

    # ΔMAP: rises moderately with exercise [AHA Circulation exercise standards]
    delta_map = (0.8 + 1.5 * intensity) * intensity

    # ΔCO: proportional to intensity; SV index modulates stroke contribution
    delta_co  = (2.0 + 2.5 * intensity) * intensity * state.sv_index

    return delta_hr, delta_map, delta_co, intensity


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Orthostatic Deconditioning (Research-Corrected)
# ─────────────────────────────────────────────────────────────────────────────

def _orthostatic_deconditioning(
    t_since_landing: float,
    state: PhysioState,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """
    Cardiovascular impact of returning to 1 G after microgravity.

    Research basis:
    - HR ratio = 1.3 × pre-flight on landing day [R6].
      For a baseline HR of 72 bpm → landing day standing HR ≈ 93–94 bpm.
      ΔHR_peak ≈ +22 bpm relative to supine pre-flight baseline.
    - Plasma volume depleted 7–20% pre-flight (we model 12%) [R7].
    - Post-landing PRIMARY haemodynamic event: postural SV decrease [R8].
      Compensatory tachycardia maintains CO; MAP is protected if vasoconstriction
      is intact — absent in symptomatic OI astronauts [R8].
    - Sympathetic nerve activity ~60% above pre-flight baseline on landing day [R12].

    Model: Dual-exponential recovery [R6]:
      Fast component (τ₁ = 18 min):  autonomic cardiovascular reflex
      Slow component (τ₂ = 280 min): plasma volume restoration [R7]
      Ratio: 55% fast / 45% slow (empirically tuned to match HR ratio data [R6])

    OI severity scales with microgravity exposure via log relationship:
      oi_mag = clamp(log(1 + µg_duration_hours) / log(1 + 4), 0, 1)
      This gives ~0.5 at 2 hours, ~1.0 at 4+ hours exposure [R6, R8].

    Args:
        t_since_landing:  Minutes since touchdown (0 at splashdown).
        state:            Current PhysioState.
        rng:              RNG for noise.

    Returns:
        (delta_hr, delta_map, delta_co)
    """
    mg_hours = state.microgravity_exposure_min / 60.0
    # Log-saturation with 4-hour reference point [R6, R8]
    oi_mag = float(np.clip(np.log1p(mg_hours) / np.log1p(4.0), 0.0, 1.0))

    # Plasma volume severity modifier: more loss → more OI [R7, R13]
    pv_severity = float(np.clip((1.0 - state.plasma_volume_frac) / 0.12, 0.5, 1.5))

    fast_decay = np.exp(-t_since_landing / 18.0)    # τ₁ = 18 min [R6]
    slow_decay = np.exp(-t_since_landing / 280.0)   # τ₂ = 280 min [R7]

    # ΔHR: peak +22 bpm at landing [derived from HR ratio 1.3 × 72 bpm, R6]
    peak_hr_rise = 22.0 * oi_mag * pv_severity
    delta_hr = peak_hr_rise * (0.55 * fast_decay + 0.45 * slow_decay)

    # ΔMAP: depends on vasoconstriction adequacy [R8]
    # Finishers (adequate vasoconstriction): MAP protected (small drop).
    # Non-finishers: MAP falls significantly.
    # We model average astronaut (some vasoconstriction maintained):
    peak_map_drop = 10.0 * oi_mag * pv_severity * (2.0 - state.sympathetic_index)
    delta_map = -peak_map_drop * (0.45 * fast_decay + 0.55 * slow_decay)

    # ΔCO: SV falls postally; HR compensates → CO maintained [R8]
    sv_drop = 0.30 * oi_mag * pv_severity
    peak_co_drop = sv_drop * 5.2   # scale to L/min
    delta_co = -peak_co_drop * fast_decay * 0.6   # HR partially compensates

    return delta_hr, delta_map, delta_co


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Physiological Signal Smoother
# ─────────────────────────────────────────────────────────────────────────────

def _smooth_signal(arr: np.ndarray, fs_hz: float = 1 / 300.0, cutoff_hz: float = 0.002) -> np.ndarray:
    """
    Apply a low-pass Butterworth filter to remove unphysiological step-changes.

    Physiological cardiovascular signals do not change instantaneously between
    mission phases. A 4th-order Butterworth filter with a ~0.002 Hz cutoff
    (characteristic time ~500 s = 8.3 min) ensures smooth phase transitions.
    Uses scipy.signal.butter + sosfiltfilt (zero-phase, no lag).

    Args:
        arr:       1-D array of values (HR, MAP, or CO).
        fs_hz:     Sampling frequency in Hz (default: 1 sample per 5 min).
        cutoff_hz: Cut-off frequency in Hz (default: 0.002 Hz ≈ 8-min period).

    Returns:
        Smoothed array (same shape as arr).
    """
    if len(arr) < 9:
        return arr
    nyq = 0.5 * fs_hz
    norm_cutoff = cutoff_hz / nyq
    if norm_cutoff >= 1.0:
        return arr
    sos = sp_signal.butter(4, norm_cutoff, btype="low", output="sos")
    return sp_signal.sosfiltfilt(sos, arr)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — Main Simulation
# ─────────────────────────────────────────────────────────────────────────────

def simulate_mission(
    base_df: pd.DataFrame,
    seed: int = 42,
    freq_minutes: int = 5,
    astronaut_g_tolerance: float = 1.0,
    decon_severity: float = 1.0,
    t_reentry_start: float = 150.0,
    microgravity_start: float = 22.0,
    apply_smoothing: bool = True,
) -> pd.DataFrame:
    """
    Simulate full-mission cardiovascular physiology with research-grounded values.

    ── Phase Timeline (minutes from T-0 launch) ──────────────────────────────
      0 – 22 min   Launch ascent     +Gz: 1 → 4 → 0 G  [R2]
     22 – 30 min   Staging / coast   Gz ≈ 0 G
     30 – 90 min   Microgravity      Fluid shift; CO +18–41%; MAP stable [R3, R4]
     90 – 150 min  Exercise          In-orbit protocol; gamma-distributed intensity [R14]
    150 – 180 min  Re-entry          +Gx: 0 → 4 Gx; CV-equiv Gz ≈ Gx × 0.55 [R9, R10]
    180 +    min   Recovery          1-G return; OI; dual-exp recovery [R6, R7, R8]

    ── Key Research Corrections vs v1 ────────────────────────────────────────
    1. k_hr = 10 bpm/G (research value [R1]) — was 8 bpm/G.
    2. Peak launch G = 4.0 G per LVM3 human-rating limit [R2] — was 3.6 G.
    3. Microgravity MAP: stable (noise only), NOT systematically reduced [R4].
    4. Microgravity CO: modelled via SV_index state variable (SV +35–46%) [R3].
    5. Exercise: Gamma-distributed intensity (more realistic than lognormal).
    6. Re-entry: Gx peak 4.0 G Soyuz-class [R9]; CV-equiv factor 0.55 [R10].
    7. Orthostatic: HR ratio 1.3 × baseline on landing day [R6]; dual-exp τ.
    8. Baroreflex: BRS 10.4 → 18.3 ms/mmHg early flight [R5]; state-tracked.
    9. Plasma volume: 7–20% loss (model: 12%) [R7]; feeds into CO, MAP, OI.
   10. Post-processing: Butterworth low-pass filter for smooth phase transitions.

    Args:
        base_df:              BioGears-style DataFrame.
                              Required cols: time_min, heart_rate_bpm,
                              mean_arterial_pressure_mmHg, cardiac_output_L_min.
        seed:                 RNG seed.
        freq_minutes:         (API-compatibility; unused internally.)
        astronaut_g_tolerance: AGSM/training factor. 1.0 = average trained [R1].
        decon_severity:       OI severity multiplier (1.0 = average astronaut).
        t_reentry_start:      Minute at which re-entry begins (default 150).
        microgravity_start:   Minute of orbit insertion / 0-G onset (default 22).
        apply_smoothing:      If True, Butterworth-smooth HR, MAP, CO post-sim.

    Returns:
        DataFrame with new columns:
        gz_profile, gx_profile, cv_equiv_gz, g_onset_rate_G_per_min,
        plasma_volume_frac, baroreflex_sensitivity_ms_mmhg, sv_index,
        sympathetic_index, systolic_bp_mmHg, diastolic_bp_mmHg,
        cardiac_power_output_w, rate_pressure_product,
        mission_phase, event_marker, high_g_risk_flag, exercise_intensity
    """
    rng = np.random.default_rng(seed)
    df  = base_df.copy().sort_values("time_min").reset_index(drop=True)

    # Poisson-distributed random stress events across full mission [prior work]
    duration_hours = (df["time_min"].max() - df["time_min"].min()) / 60.0
    stress_events  = poisson_stress_events(rate_per_hour=0.6, duration_hours=duration_hours, rng=rng)

    # Pre-compute G-force arrays
    times           = df["time_min"].astype(float).values
    gz_launch_arr   = np.array([_gz_launch(t)                         for t in times])
    gx_reentry_arr  = np.array([_gx_reentry(t, t_reentry_start)       for t in times])
    cv_gz_arr       = np.array([_gx_to_cv_equiv_gz(g)                 for g in gx_reentry_arr])

    # G-onset rate: first derivative of active Gz (G/min)
    dt_arr = np.maximum(np.gradient(times), 1e-3)
    # Launch phase: d(gz_launch)/dt; re-entry phase: d(cv_gz)/dt
    onset_launch  = np.abs(np.gradient(gz_launch_arr) / dt_arr)
    onset_reentry = np.abs(np.gradient(cv_gz_arr)     / dt_arr)

    # Initialise physiological state
    state = PhysioState()

    # Output column initialisation
    for col in [
        "gz_profile", "gx_profile", "cv_equiv_gz", "g_onset_rate_G_per_min",
        "plasma_volume_frac", "baroreflex_sensitivity_ms_mmhg", "sv_index",
        "sympathetic_index", "systolic_bp_mmHg", "diastolic_bp_mmHg",
        "cardiac_power_output_w", "rate_pressure_product",
        "exercise_intensity",
    ]:
        df[col] = 0.0
    for col in ("mission_phase", "event_marker"):
        df[col] = ""
    df["high_g_risk_flag"] = False

    # ── Time-step loop ────────────────────────────────────────────────────────
    for i, t in enumerate(times):
        hr   = float(df.loc[i, "heart_rate_bpm"])
        map_ = float(df.loc[i, "mean_arterial_pressure_mmHg"])
        co   = float(df.loc[i, "cardiac_output_L_min"])
        ex_intensity = 0.0

        gz_now = gz_launch_arr[i]
        gx_now = gx_reentry_arr[i]
        cv_gz  = cv_gz_arr[i]

        # Determine active G-axis and onset rate
        in_reentry = t_reentry_start <= t < t_reentry_start + 30.0
        active_gz  = cv_gz if in_reentry else gz_now
        onset_rate = onset_reentry[i] if in_reentry else onset_launch[i]

        # Time step for state update
        dt_min = float(dt_arr[i])

        # ── Phase dispatch ────────────────────────────────────────────────────
        if t < microgravity_start:
            phase  = "launch_ascent"
            marker = f"Launch Gz={gz_now:.2f} G"
            state  = _update_physio_state(state, phase, dt_min)
            dhr, dmap, dco = _g_physiology(gz_now, onset_rate, state, astronaut_g_tolerance)
            hr   += dhr
            map_ += dmap
            co   += dco

        elif microgravity_start <= t < 30.0:
            phase  = "staging_coast"
            marker = "Staging / coast (≈0 G)"
            state  = _update_physio_state(state, "microgravity_fluid_shift", dt_min)
            # Very low residual G — only onset-rate effect, no sustained load
            dhr, dmap, dco = _g_physiology(gz_now, onset_rate, state, astronaut_g_tolerance)
            hr   += dhr
            map_ += dmap
            co   += dco

        elif 30.0 <= t < 90.0:
            phase      = "microgravity_fluid_shift"
            phase_time = t - 30.0
            marker     = "Microgravity / cephalad fluid shift"
            state      = _update_physio_state(state, phase, dt_min)
            dhr, dmap, dco = _microgravity_delta(phase_time, state, rng)
            hr   += dhr
            map_ += dmap
            co   += dco

        elif 90.0 <= t < t_reentry_start:
            phase  = "exercise_workload"
            marker = "In-orbit exercise protocol"
            state  = _update_physio_state(state, phase, dt_min)
            dhr, dmap, dco, ex_intensity = _exercise_delta(rng, state)
            hr   += dhr
            map_ += dmap
            co   += dco

        elif in_reentry:
            phase  = "reentry_stress"
            marker = f"Re-entry Gx={gx_now:.2f} G  (CV-equiv={cv_gz:.2f} G)"
            state  = _update_physio_state(state, phase, dt_min)
            dhr, dmap, dco = _g_physiology(cv_gz, onset_rate, state, astronaut_g_tolerance)
            hr   += dhr
            map_ += dmap
            co   += dco

        else:
            phase           = "recovery"
            t_land          = t_reentry_start + 30.0
            t_since_landing = max(0.0, t - t_land)
            marker          = "Recovery / orthostatic readaptation"
            state           = _update_physio_state(state, phase, dt_min)
            dhr, dmap, dco  = _orthostatic_deconditioning(t_since_landing, state, rng)
            # Apply decon_severity multiplier
            hr   += dhr * decon_severity
            map_ += dmap * decon_severity
            co   += dco * decon_severity

        # ── Biological noise ─────────────────────────────────────────────────
        hr   += normal_variation(0, 1.8, 1, rng)[0]
        map_ += normal_variation(0, 1.2, 1, rng)[0]
        co   += normal_variation(0, 0.08, 1, rng)[0]

        # ── Poisson stress bursts ────────────────────────────────────────────
        if stress_events > 0 and rng.random() < (stress_events / max(len(df), 1)):
            hr   += rng.uniform(3, 8)
            map_ -= rng.uniform(1, 4)

        # ── Derived quantities ───────────────────────────────────────────────
        sbp  = map_ + 18.0 + rng.normal(0, 1.8)
        dbp  = map_ - 10.0 + rng.normal(0, 1.2)

        hr   = float(bounded(hr,   lower=35,  upper=190))
        map_ = float(bounded(map_, lower=40,  upper=130))
        co   = float(bounded(co,   lower=2.0, upper=12.0))
        sbp  = float(bounded(sbp,  lower=70,  upper=220))
        dbp  = float(bounded(dbp,  lower=30,  upper=150))

        # Cardiac Power Output [W] = MAP × CO × 0.0022  (unit conversion factor)
        cpo = 0.0022 * map_ * co
        # Rate-Pressure Product (myocardial O₂ demand proxy) [×10⁻³]
        rpp = hr * sbp / 1000.0

        # High-G risk: effective Gz > 3.5 (pre-G-LOC territory, see Section 3)
        high_g = active_gz > 3.5

        # Write row
        df.loc[i, "heart_rate_bpm"]                    = hr
        df.loc[i, "mean_arterial_pressure_mmHg"]       = map_
        df.loc[i, "cardiac_output_L_min"]              = co
        df.loc[i, "systolic_bp_mmHg"]                  = sbp
        df.loc[i, "diastolic_bp_mmHg"]                 = dbp
        df.loc[i, "cardiac_power_output_w"]            = cpo
        df.loc[i, "rate_pressure_product"]             = rpp
        df.loc[i, "gz_profile"]                        = gz_now
        df.loc[i, "gx_profile"]                        = gx_now
        df.loc[i, "cv_equiv_gz"]                       = active_gz
        df.loc[i, "g_onset_rate_G_per_min"]            = onset_rate
        df.loc[i, "plasma_volume_frac"]                = state.plasma_volume_frac
        df.loc[i, "baroreflex_sensitivity_ms_mmhg"]    = state.baroreflex_sensitivity_ms_mmhg
        df.loc[i, "sv_index"]                          = state.sv_index
        df.loc[i, "sympathetic_index"]                 = state.sympathetic_index
        df.loc[i, "exercise_intensity"]                = ex_intensity
        df.loc[i, "mission_phase"]                     = phase
        df.loc[i, "event_marker"]                      = marker
        df.loc[i, "high_g_risk_flag"]                  = high_g

    # ── Post-processing: smooth step-changes at phase boundaries ──────────────
    if apply_smoothing:
        fs = 1.0 / (float(np.median(np.diff(times))) * 60.0)   # Hz
        for col in ("heart_rate_bpm", "mean_arterial_pressure_mmHg", "cardiac_output_L_min"):
            df[col] = _smooth_signal(df[col].values, fs_hz=fs, cutoff_hz=0.0015)
            # Re-clip after smoothing to maintain physiological bounds
        df["heart_rate_bpm"]                 = df["heart_rate_bpm"].clip(35, 190)
        df["mean_arterial_pressure_mmHg"]    = df["mean_arterial_pressure_mmHg"].clip(40, 130)
        df["cardiac_output_L_min"]           = df["cardiac_output_L_min"].clip(2.0, 12.0)

    return df
