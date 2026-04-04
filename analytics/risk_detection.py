from __future__ import annotations

import numpy as np
import pandas as pd


HYPOTENSION_MAP_THRESHOLD = 65.0
TACHYCARDIA_HR_THRESHOLD = 100.0


def detect_risks(df: pd.DataFrame, instability_index: float, instability_threshold: float = 0.15) -> pd.DataFrame:
    """Add risk flags to the dataframe."""
    out = df.copy()
    out["hypotension_risk"] = out["mean_arterial_pressure_mmHg"] < HYPOTENSION_MAP_THRESHOLD
    out["tachycardia"] = out["heart_rate_bpm"] > TACHYCARDIA_HR_THRESHOLD
    out["instability_risk"] = False

    # Instability is stored as a global mission-level flag, but we mirror it per row for dashboarding.
    if instability_index >= instability_threshold:
        out["instability_risk"] = True

    out["risk_score"] = (
        out["hypotension_risk"].astype(int)
        + out["tachycardia"].astype(int)
        + out["instability_risk"].astype(int)
    )
    return out


def summarize_risks(df: pd.DataFrame) -> dict:
    """Return a compact risk summary."""
    return {
        "hypotension_count": int(df["hypotension_risk"].sum()),
        "tachycardia_count": int(df["tachycardia"].sum()),
        "any_risk_points": int((df["risk_score"] > 0).sum()),
    }
