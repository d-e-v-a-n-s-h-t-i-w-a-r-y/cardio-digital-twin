from __future__ import annotations

import pandas as pd


def estimate_recovery_time(
    df: pd.DataFrame,
    stress_event_name: str = "reentry_stress",
    baseline_window: int = 6,
    tolerance_fraction: float = 0.10,
) -> float | None:
    """Estimate the recovery time after the last stress event.

    Recovery is defined as the first time after the stress event where:
    - HR returns within `tolerance_fraction` of baseline HR, and
    - MAP returns within `tolerance_fraction` of baseline MAP
    for a continuous window of `baseline_window` samples.

    The function checks both ``event`` (raw BioGears CSV column) and
    ``event_marker`` / ``mission_phase`` (simulated event columns produced by
    ``event_simulator.simulate_mission``).  This makes it work transparently
    whether the caller uses real BioGears data or synthetic simulation data.

    Parameters
    ----------
    df:
        DataFrame with at least columns:
        - time_min
        - heart_rate_bpm
        - mean_arterial_pressure_mmHg
        and *one of*:
        - event          (raw BioGears CSV)
        - event_marker   (synthetic simulation output)
        - mission_phase  (synthetic simulation output)
    stress_event_name:
        The value to look for in the event column to mark the end of stress.
        Defaults to ``"reentry_stress"``.
    baseline_window:
        Minimum number of consecutive samples within bounds to declare recovery.
    tolerance_fraction:
        Fractional tolerance for HR and MAP to be considered "recovered"
        (e.g. 0.10 = within 10 % of baseline).

    Returns
    -------
    float or None
        Recovery time in minutes after the last stress row, or None if the
        astronaut did not recover within the observation window.
    """
    # ------------------------------------------------------------------
    # Detect which column holds event labels (handles both data sources)
    # ------------------------------------------------------------------
    event_col: str | None = None
    for candidate in ("event", "event_marker", "mission_phase"):
        if candidate in df.columns:
            event_col = candidate
            break

    if event_col is None:
        return None

    df = df.sort_values("time_min").reset_index(drop=True)
    stress_rows = df.index[df[event_col] == stress_event_name].tolist()

    # For simulated data the event_marker column uses a different label; try a
    # fallback that matches the "Re-entry stress" marker from event_simulator.
    if not stress_rows and event_col in ("event_marker", "mission_phase"):
        fallback_labels = {
            "event_marker":   "Re-entry stress",
            "mission_phase":  "reentry_stress",
        }
        fallback = fallback_labels.get(event_col)
        if fallback:
            stress_rows = df.index[df[event_col] == fallback].tolist()

    if not stress_rows:
        return None

    stress_end_idx = stress_rows[-1]
    if stress_end_idx + 1 >= len(df):
        return None

    # Baseline is computed from the first few rows *before* the stress event.
    baseline = df.iloc[: max(1, min(stress_end_idx, baseline_window))]
    baseline_hr  = float(baseline["heart_rate_bpm"].mean())
    baseline_map = float(baseline["mean_arterial_pressure_mmHg"].mean())

    hr_bounds  = (baseline_hr  * (1 - tolerance_fraction), baseline_hr  * (1 + tolerance_fraction))
    map_bounds = (baseline_map * (1 - tolerance_fraction), baseline_map * (1 + tolerance_fraction))

    post = df.iloc[stress_end_idx + 1:].copy()
    if post.empty:
        return None

    ok = (
        post["heart_rate_bpm"].between(*hr_bounds)
        & post["mean_arterial_pressure_mmHg"].between(*map_bounds)
    ).astype(int)

    streak = 0
    for idx, good in zip(post.index, ok):
        streak = streak + 1 if good else 0
        if streak >= baseline_window:
            return float(df.loc[idx, "time_min"] - df.loc[stress_end_idx, "time_min"])

    return None
