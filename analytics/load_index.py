from __future__ import annotations

import pandas as pd


def cardiovascular_workload_index(df: pd.DataFrame) -> pd.Series:
    """Compute the cardiovascular workload index = HR × Cardiac Output."""
    return df["heart_rate_bpm"].astype(float) * df["cardiac_output_L_min"].astype(float)
