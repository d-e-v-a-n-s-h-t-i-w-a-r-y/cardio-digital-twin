
from __future__ import annotations

import numpy as np
import pandas as pd

from analytics.instability import cardiovascular_instability_index
from analytics.recovery import estimate_recovery_time
from analytics.risk_detection import detect_risks


def run_monte_carlo(
    base_df: pd.DataFrame,
    n_runs: int = 100,
    hr_noise_sd: float = 2.0,
    map_noise_sd: float = 1.5,
    co_noise_sd: float = 0.12,
    seed: int = 123,
) -> dict:
    """Run Monte Carlo simulations around a base mission profile."""
    rng = np.random.default_rng(seed)

    hypotension_hits = 0
    tachycardia_hits = 0
    instability_hits = 0
    recovery_times: list[float] = []

    for _ in range(n_runs):
        df = base_df.copy()
        df['heart_rate_bpm'] = np.clip(df['heart_rate_bpm'] + rng.normal(0, hr_noise_sd, size=len(df)), 35, 190)
        df['mean_arterial_pressure_mmHg'] = np.clip(df['mean_arterial_pressure_mmHg'] + rng.normal(0, map_noise_sd, size=len(df)), 40, 130)
        df['cardiac_output_L_min'] = np.clip(df['cardiac_output_L_min'] + rng.normal(0, co_noise_sd, size=len(df)), 2.0, 12.0)

        cii, _ = cardiovascular_instability_index(df)
        analyzed = detect_risks(df, cii)
        hypotension_hits += int(analyzed['hypotension_risk'].any())
        tachycardia_hits += int(analyzed['tachycardia'].any())
        instability_hits += int(cii >= 0.15)

        rt = estimate_recovery_time(analyzed)
        if rt is not None:
            recovery_times.append(rt)

    return {
        'n_runs': int(n_runs),
        'hypotension_probability': hypotension_hits / n_runs,
        'tachycardia_probability': tachycardia_hits / n_runs,
        'instability_probability': instability_hits / n_runs,
        'mean_recovery_time_min': float(np.mean(recovery_times)) if recovery_times else None,
        'recovery_samples': recovery_times,
    }
