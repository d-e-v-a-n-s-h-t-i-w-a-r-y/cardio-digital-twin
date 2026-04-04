from __future__ import annotations

import numpy as np
import pandas as pd

from simulation.probabilistic_inputs import normal_variation, poisson_stress_events, lognormal_exercise_intensity, bounded


def simulate_mission(
    base_df: pd.DataFrame,
    seed: int = 42,
    freq_minutes: int = 5,
) -> pd.DataFrame:
    """Simulate mission physiology from a baseline BioGears-style dataframe.

    The simulation uses the baseline shape as a template and adds:
    - microgravity fluid shift
    - exercise workload
    - re-entry stress
    - recovery dynamics
    - random stress events
    """
    rng = np.random.default_rng(seed)
    df = base_df.copy().sort_values("time_min").reset_index(drop=True)

    # Random stress event count using Poisson model.
    duration_hours = (df["time_min"].max() - df["time_min"].min()) / 60.0
    stress_events = poisson_stress_events(rate_per_hour=0.6, duration_hours=duration_hours, rng=rng)

    df["mission_phase"] = "baseline"
    df["event_marker"] = ""

    for i, t in enumerate(df["time_min"].astype(float)):
        hr = float(df.loc[i, "heart_rate_bpm"])
        map_ = float(df.loc[i, "mean_arterial_pressure_mmHg"])
        co = float(df.loc[i, "cardiac_output_L_min"])

        if 30 <= t < 90:
            phase = "microgravity_fluid_shift"
            hr += 3.0 + 0.06 * (t - 30)
            map_ -= 2.0 + 0.05 * (t - 30)
            co += 0.03 * (t - 30)
            marker = "Fluid shift"
        elif 90 <= t < 150:
            phase = "exercise_workload"
            intensity = float(lognormal_exercise_intensity(mu=0.35, sigma=0.20, size=1, rng=rng)[0])
            hr += 10 + 12 * intensity
            map_ += 0.8 * intensity
            co += 0.5 * intensity
            marker = "Exercise"
        elif 150 <= t < 180:
            phase = "reentry_stress"
            hr += 20 + rng.normal(0, 3)
            map_ -= 6 + rng.normal(0, 2)
            co += 0.7 + rng.normal(0, 0.15)
            marker = "Re-entry"
        else:
            phase = "recovery"
            recovery_factor = max(0.0, (240 - t) / 60.0)
            hr -= 2.0 * recovery_factor
            map_ += 1.8 * recovery_factor
            co -= 0.15 * recovery_factor
            marker = "Recovery" if t >= 180 else ""

        # Add physiologic variability.
        hr = hr + normal_variation(0, 1.8, 1, rng)[0]
        map_ = map_ + normal_variation(0, 1.2, 1, rng)[0]
        co = co + normal_variation(0, 0.08, 1, rng)[0]

        # Random stress bursts as short spikes/dips.
        if stress_events > 0 and rng.random() < (stress_events / max(len(df), 1)):
            hr += rng.uniform(3, 8)
            map_ -= rng.uniform(1, 4)

        sbp = map_ + 18 + rng.normal(0, 1.8)
        dbp = map_ - 10 + rng.normal(0, 1.2)

        df.loc[i, "heart_rate_bpm"] = float(bounded(hr, lower=35, upper=190))
        df.loc[i, "mean_arterial_pressure_mmHg"] = float(bounded(map_, lower=40, upper=130))
        df.loc[i, "cardiac_output_L_min"] = float(bounded(co, lower=2.0, upper=12.0))
        df.loc[i, "systolic_bp_mmHg"] = float(bounded(sbp, lower=70, upper=220))
        df.loc[i, "diastolic_bp_mmHg"] = float(bounded(dbp, lower=30, upper=150))
        df.loc[i, "mission_phase"] = phase
        df.loc[i, "event_marker"] = marker

    # Add a few explicit event labels for timeline visualization.
    def mark_range(start, end, label):
        mask = (df["time_min"] >= start) & (df["time_min"] < end)
        df.loc[mask, "event_marker"] = label

    mark_range(30, 90, "Microgravity fluid shift")
    mark_range(90, 150, "Exercise workload")
    mark_range(150, 180, "Re-entry stress")
    mark_range(180, 241, "Recovery phase")

    return df
