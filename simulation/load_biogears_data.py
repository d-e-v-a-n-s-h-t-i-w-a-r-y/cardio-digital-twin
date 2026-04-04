from __future__ import annotations

from pathlib import Path
import pandas as pd


EXPECTED_COLUMNS = [
    "time_min",
    "event",
    "heart_rate_bpm",
    "mean_arterial_pressure_mmHg",
    "systolic_bp_mmHg",
    "diastolic_bp_mmHg",
    "cardiac_output_L_min",
]


def load_biogears_data(csv_path: str | Path) -> pd.DataFrame:
    """Load a BioGears-style cardiovascular CSV file.

    Parameters
    ----------
    csv_path:
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with expected columns.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    df = df[EXPECTED_COLUMNS].copy()
    df["time_min"] = pd.to_numeric(df["time_min"], errors="coerce")
    for col in EXPECTED_COLUMNS[2:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().sort_values("time_min").reset_index(drop=True)
    return df
