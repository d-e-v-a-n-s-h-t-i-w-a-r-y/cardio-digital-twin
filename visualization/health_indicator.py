from __future__ import annotations


def health_status(summary: dict, instability_threshold: float = 0.15) -> tuple[str, str]:
    """Return a traffic-light status and a simple text label."""
    hypotension = summary.get("hypotension_count", 0)
    tachycardia = summary.get("tachycardia_count", 0)
    instability = summary.get("instability_index", 0.0)

    if hypotension > 0 or tachycardia > 0 or instability >= instability_threshold:
        if hypotension > 2 or tachycardia > 2 or instability >= instability_threshold * 1.5:
            return "Red", "High cardiovascular risk"
        return "Yellow", "Moderate cardiovascular risk"
    return "Green", "Stable cardiovascular state"
