from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from analytics.instability import cardiovascular_instability_index
from analytics.load_index import cardiovascular_workload_index
from analytics.recovery import estimate_recovery_time
from analytics.risk_detection import detect_risks, summarize_risks


@dataclass
class DigitalTwinResult:
    data: pd.DataFrame
    instability_index: float
    instability_details: dict
    recovery_time_min: float | None
    summary: dict


class CardiovascularDigitalTwin:
    """High-level wrapper that combines simulation and analytics."""

    def __init__(self, base_df: pd.DataFrame):
        self.base_df = base_df.copy()

    def analyze(self, simulated_df: pd.DataFrame) -> DigitalTwinResult:
        instability_index, details = cardiovascular_instability_index(simulated_df)
        analyzed = detect_risks(simulated_df, instability_index)
        analyzed["cardiovascular_workload_index"] = cardiovascular_workload_index(analyzed)
        recovery_time = estimate_recovery_time(analyzed)
        summary = summarize_risks(analyzed)
        summary["instability_index"] = instability_index
        summary["recovery_time_min"] = recovery_time
        summary["mean_workload_index"] = float(analyzed["cardiovascular_workload_index"].mean())
        return DigitalTwinResult(
            data=analyzed,
            instability_index=instability_index,
            instability_details=details,
            recovery_time_min=recovery_time,
            summary=summary,
        )
