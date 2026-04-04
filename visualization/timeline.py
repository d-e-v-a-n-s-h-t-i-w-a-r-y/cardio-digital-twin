from __future__ import annotations

import plotly.graph_objects as go
import pandas as pd


def add_timeline_markers(fig: go.Figure, df: pd.DataFrame) -> go.Figure:
    """Add event markers to a Plotly time-series figure."""
    if "event_marker" not in df.columns:
        return fig

    events = df.loc[df["event_marker"].astype(str).str.len() > 0, ["time_min", "event_marker"]].drop_duplicates()
    for _, row in events.iterrows():
        fig.add_vline(x=row["time_min"], line_width=1, line_dash="dash", line_color="gray")
        fig.add_annotation(
            x=row["time_min"],
            y=1.02,
            xref="x",
            yref="paper",
            text=str(row["event_marker"]),
            showarrow=False,
            font=dict(size=10),
        )
    return fig
