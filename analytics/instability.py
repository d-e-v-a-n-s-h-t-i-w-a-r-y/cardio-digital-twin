from __future__ import annotations

import numpy as np
import pandas as pd


def cardiovascular_instability_index(df: pd.DataFrame) -> tuple[float, dict]:
    """Compute a simple cardiovascular instability index.

    The index combines normalized variability in HR and MAP:
        index = (std(HR) / mean(HR)) + (std(MAP) / mean(MAP))

    Returns
    -------
    (index, details)
    """
    hr = df["heart_rate_bpm"].astype(float)
    map_ = df["mean_arterial_pressure_mmHg"].astype(float)

    hr_mean = float(hr.mean())
    map_mean = float(map_.mean())
    hr_std = float(hr.std(ddof=0))
    map_std = float(map_.std(ddof=0))

    index = (hr_std / max(hr_mean, 1e-6)) + (map_std / max(map_mean, 1e-6))
    details = {
        "hr_mean": hr_mean,
        "map_mean": map_mean,
        "hr_std": hr_std,
        "map_std": map_std,
    }
    return float(index), details
