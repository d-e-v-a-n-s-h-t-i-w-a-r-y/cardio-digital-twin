"""
Astronaut Cardiovascular Digital Twin — Streamlit Dashboard
============================================================
BioGears-inspired cardiovascular simulation for space mission risk monitoring.
ISRO Space Medicine — Gaganyaan Mission Support

Run:
    streamlit run visualization/dashboard.py

Two operating modes
-------------------
1. Simulation mode  — synthetic physiology generated from sidebar inputs.
2. BioGears CSV mode — upload a ``cardiovascular_data.csv`` file that matches
   the project's expected schema:
       time_min, event, heart_rate_bpm, mean_arterial_pressure_mmHg,
       systolic_bp_mmHg, diastolic_bp_mmHg, cardiac_output_L_min
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Cardio Digital Twin — ISRO Gaganyaan Monitor",
    page_icon="🛸",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS — forced dark mode + ISRO premium aesthetic
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    /* ── FORCE DARK MODE regardless of system/Streamlit theme ── */
    html, body {
        background-color: #060e18 !important;
        color: #d0e8f5 !important;
    }
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    section[data-testid="stSidebar"],
    .stApp {
        background-color: #060e18 !important;
        color: #d0e8f5 !important;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1628 0%, #060e18 100%) !important;
        border-right: 1px solid #1e3a5f !important;
    }
    /* Sidebar text */
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {
        color: #b0cfe8 !important;
    }
    /* Input widgets */
    [data-testid="stNumberInput"] input,
    [data-testid="stSelectbox"] div,
    [data-testid="stSlider"] { color: #d0e8f5 !important; }
    /* Tabs */
    [data-testid="stTabs"] button {
        color: #7fb3d3 !important;
        background: transparent !important;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #4fc3f7 !important;
        border-bottom: 2px solid #4fc3f7 !important;
    }
    /* Divider */
    hr { border-color: #1e3a5f !important; }
    /* Dataframe */
    [data-testid="stDataFrame"] { background: #0a1628 !important; }
    /* Alert/info boxes */
    [data-testid="stAlert"] { background-color: #0a1628 !important; border-color: #1e3a5f !important; }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Header gradient — ISRO saffron-orange-navy */
    .dashboard-header {
        background: linear-gradient(135deg, #0a1628 0%, #12243d 50%, #1a1000 100%);
        border-radius: 12px;
        padding: 24px 32px;
        margin-bottom: 20px;
        border: 1px solid #2a4a7f;
        box-shadow: 0 4px 24px rgba(0,0,0,0.5);
    }
    .dashboard-header h1 {
        color: #f4c842;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .dashboard-header .subtitle {
        color: #f4a240;
        margin: 2px 0 4px 0;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }
    .dashboard-header p {
        color: #7fb3d3;
        margin: 4px 0 0 0;
        font-size: 0.88rem;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #0d1b2a, #142130);
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        padding: 14px 18px;
    }
    [data-testid="stMetricLabel"]  { color: #7fb3d3 !important; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; }
    [data-testid="stMetricValue"]  { color: #e8f4fd !important; font-size: 1.6rem; font-weight: 700; }
    [data-testid="stMetricDelta"]  { font-size: 0.75rem; }

    /* Section header — ISRO orange accent */
    .section-title {
        font-size: 1rem;
        font-weight: 600;
        color: #f4a240;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 16px 0 8px 0;
        padding-bottom: 6px;
        border-bottom: 1px solid #f4a24040;
    }

    /* Graph caption */
    .graph-caption {
        background: #060e18;
        border-left: 3px solid #f4a240;
        border-radius: 0 6px 6px 0;
        padding: 8px 14px;
        color: #94b8cc;
        font-size: 0.8rem;
        margin-bottom: 8px;
        font-style: italic;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="dashboard-header">
      <h1>🛸 Astronaut Cardiovascular Digital Twin</h1>
      <p class="subtitle">ISRO · Gaganyaan Human Spaceflight Programme · Space Medicine Division</p>
      <p>BioGears-inspired real-time cardiovascular risk monitor across mission phases</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — mode selection + inputs
# ---------------------------------------------------------------------------
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bd/Indian_Space_Research_Organisation_Logo.svg/800px-Indian_Space_Research_Organisation_Logo.svg.png",
    width=110,
)
st.sidebar.markdown("## Mission Control Inputs")

data_mode = st.sidebar.radio("Data Source", ["🔬 Simulation Mode", "📂 BioGears CSV Upload"])

df: pd.DataFrame | None = None
source_label = ""

# ── BioGears CSV upload ──────────────────────────────────────────────────────
if data_mode == "📂 BioGears CSV Upload":
    uploaded = st.sidebar.file_uploader(
        "Upload any project CSV",
        type=["csv"],
        help=(
            "Accepts all CSV formats in this project:\n"
            "• cardiovascular_data.csv\n"
            "• final_digital_twin_dataset.csv\n"
            "• biogears_real_data.csv\n"
            "Column names are auto-detected."
        ),
    )

    if uploaded is not None:
        try:
            raw = pd.read_csv(io.StringIO(uploaded.read().decode("utf-8")))
            cols = raw.columns.tolist()

            # ── Auto-detect column mappings ───────────────────────────────
            # Each entry: internal_name -> list of candidate column names (case-insensitive)
            CANDIDATES = {
                "time": [
                    "time_min", "Time_hr", "Time(s)", "time", "Time",
                    "timestamp", "t", "minutes", "hours",
                ],
                "heart_rate": [
                    "heart_rate_bpm", "HeartRate", "HeartRate(1/min)",
                    "heart_rate", "HR", "hr",
                ],
                "map": [
                    "mean_arterial_pressure_mmHg", "MAP", "MeanArterialPressure(mmHg)",
                    "map", "Mean_Arterial_Pressure", "mean_arterial_pressure",
                ],
                "cardiac_output": [
                    "cardiac_output_L_min", "CardiacOutput", "cardiac_output",
                    "CO", "co",
                ],
                "phase": [
                    "event", "Phase", "phase", "mission_phase",
                    "event_marker", "MissionPhase", "stage",
                ],
            }

            # Build a lookup: lowercase(col) -> actual col name
            lower_to_actual = {c.lower(): c for c in cols}

            detected: dict[str, str] = {}   # internal -> actual column name
            for internal, candidates in CANDIDATES.items():
                for cand in candidates:
                    if cand.lower() in lower_to_actual:
                        detected[internal] = lower_to_actual[cand.lower()]
                        break

            # time and heart_rate and map are required
            required = ["time", "heart_rate", "map"]
            missing_req = [r for r in required if r not in detected]

            if missing_req:
                st.sidebar.error(
                    f"Could not find columns for: {missing_req}\n\n"
                    f"Your CSV has: {cols}"
                )
            else:
                # Rename detected columns to internal names
                rename_map = {v: k for k, v in detected.items()}
                df = raw.rename(columns=rename_map)

                # Convert time to minutes if it looks like hours or seconds
                time_col_original = detected["time"]
                if "hr" in time_col_original.lower() or "hour" in time_col_original.lower():
                    df["time"] = df["time"] * 60          # hours -> minutes
                elif "(s)" in time_col_original or "second" in time_col_original.lower():
                    df["time"] = df["time"] / 60          # seconds -> minutes

                # Fall back to constant cardiac output if column missing
                if "cardiac_output" not in df.columns:
                    df["cardiac_output"] = 5.2            # average default L/min

                # Fall back to "Unknown" phase if missing
                if "phase" not in df.columns:
                    df["phase"] = "Unknown"

                # Keep only what the analytics layer needs
                df = df[["time", "heart_rate", "map", "cardiac_output", "phase"]].copy()
                df = df.dropna(subset=["time", "heart_rate", "map"]).reset_index(drop=True)

                source_label = f"📂 CSV ({uploaded.name})"
                st.sidebar.success(f"✅ Loaded {len(df)} rows")

                # Show what was mapped
                with st.sidebar.expander("📋 Column mapping detected", expanded=False):
                    for internal, actual in detected.items():
                        st.write(f"`{actual}` -> **{internal}**")
                    if "cardiac_output" not in detected:
                        st.write("`cardiac_output` -> **default 5.2 L/min**")

        except Exception as exc:
            st.sidebar.error(f"Parse error: {exc}")

        # else: parse failed — error already shown in sidebar


# ── Simulation mode ──────────────────────────────────────────────────────────
if df is None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Astronaut Profile")
    profile = st.sidebar.selectbox(
        "Profile",
        ["Elite Athlete", "Average Astronaut", "Deconditioned"],
        help="Pre-sets resting HR, MAP, and cardiac output baselines.",
    )

    PROFILES = {
        "Elite Athlete":      dict(resting_hr=55, baseline_map=92, cardiac_output=6.5),
        "Average Astronaut":  dict(resting_hr=70, baseline_map=85, cardiac_output=5.2),
        "Deconditioned":      dict(resting_hr=85, baseline_map=80, cardiac_output=4.0),
    }
    p = PROFILES[profile]
    resting_hr    = st.sidebar.number_input("Resting HR (bpm)",       min_value=40,  max_value=100, value=p["resting_hr"])
    baseline_map  = st.sidebar.number_input("Baseline MAP (mmHg)",     min_value=60,  max_value=110, value=int(p["baseline_map"]))
    cardiac_output= st.sidebar.number_input("Baseline Cardiac Output", min_value=2.0, max_value=10.0,value=p["cardiac_output"], step=0.1)

    st.sidebar.markdown("### Mission Parameters")
    exercise_intensity = st.sidebar.slider("Exercise Intensity",   0.0, 1.0, 0.5, help="0 = rest, 1 = maximal effort")
    mission_duration   = st.sidebar.slider("Mission Duration (min)", 100, 400, 260)
    noise_level        = st.sidebar.slider("Physiologic Noise",    0.0, 1.0, 0.4, help="Simulated biological variability")
    mc_runs            = st.sidebar.slider("Monte Carlo Runs",      50, 500, 200, step=50)

    # ── Simulate ────────────────────────────────────────────────────────────
    rng  = np.random.default_rng(42)
    time = np.arange(mission_duration)

    hr_list, map_list, co_list, phase_list = [], [], [], []

    for t in time:
        hr_noise  = rng.normal(0, 2 * noise_level + 0.5)
        map_noise = rng.normal(0, 1.5 * noise_level + 0.3)

        if t < 40:
            phase = "Launch"
            hr    = resting_hr + 12 + hr_noise
            map_v = baseline_map + 6 + map_noise
        elif t < 120:
            phase = "Microgravity"
            hr    = resting_hr + 8 + hr_noise
            map_v = baseline_map - 5 + map_noise
        elif t < 200:
            phase = "Exercise"
            hr    = resting_hr + 50 * exercise_intensity + rng.normal(0, 3 * noise_level + 1)
            map_v = baseline_map + 3 * exercise_intensity + map_noise
        elif t < 230:
            phase = "Re-entry"
            hr    = resting_hr + 30 + rng.normal(0, 3 * noise_level + 1)
            map_v = baseline_map - 12 + map_noise
        else:
            phase = "Recovery"
            decay = (t - 230) / max(mission_duration - 230, 1)
            hr    = resting_hr + max(0, 15 * (1 - decay)) + hr_noise
            map_v = baseline_map - max(0, 6 * (1 - decay)) + map_noise

        co_val = cardiac_output + exercise_intensity * np.sin(t / 20) * 0.5

        hr_list.append(hr)
        map_list.append(map_v)
        co_list.append(co_val)
        phase_list.append(phase)

    df = pd.DataFrame({
        "time":           time,
        "heart_rate":     hr_list,
        "map":            map_list,
        "cardiac_output": co_list,
        "phase":          phase_list,
    })
    source_label = f"🔬 Simulation ({profile})"

# ===========================================================================
# ANALYTICS — shared across both data modes
# ===========================================================================
assert df is not None

# Risk flags
df["hypotension"] = df["map"] < 65
df["tachycardia"] = df["heart_rate"] > 100
df["load_index"]  = df["heart_rate"] * df["cardiac_output"]

# Cumulative Fatigue Index (Stateful Accumulator)
fatigue_vals = []
current_fatigue = 20.0
for phase in df["phase"]:
    if isinstance(phase, str):
        p_lower = phase.lower()
        if "motion" in p_lower or "microgravity" in p_lower or "shift" in p_lower:
            current_fatigue += 0.15
        elif "exercise" in p_lower or "workload" in p_lower:
            current_fatigue += 0.3
        elif "re-entry" in p_lower or "reentry" in p_lower or "stress" in p_lower:
            current_fatigue += 0.5
        elif "recovery" in p_lower:
            current_fatigue -= 0.1
        else:
            current_fatigue += 0.02
    current_fatigue = max(0.0, min(100.0, current_fatigue))
    fatigue_vals.append(current_fatigue)
df["fatigue_index"] = fatigue_vals

hypo_events  = int(df["hypotension"].sum())
tachy_events = int(df["tachycardia"].sum())

# Cardiovascular Instability Index
hr_mean  = df["heart_rate"].mean()
map_mean = df["map"].mean()
hr_std   = df["heart_rate"].std()
map_std  = df["map"].std()
cii = (hr_std / max(hr_mean, 1e-6)) + (map_std / max(map_mean, 1e-6))

# Recovery time (after last re-entry / reentry_stress row)
recovery_time: float | None = None
for stress_label in ("Re-entry", "reentry_stress", "reentry stress"):
    stress_rows = df.index[df["phase"] == stress_label].tolist()
    if stress_rows:
        last_stress = stress_rows[-1]
        base_hr  = df["heart_rate"].iloc[:6].mean()
        base_map = df["map"].iloc[:6].mean()
        streak   = 0
        for i in range(last_stress + 1, len(df)):
            hr_ok  = abs(df.loc[i, "heart_rate"] - base_hr)  < 0.10 * base_hr
            map_ok = abs(df.loc[i, "map"]         - base_map) < 0.10 * base_map
            streak = streak + 1 if (hr_ok and map_ok) else 0
            if streak >= 6:
                recovery_time = float(df.loc[i, "time"] - df.loc[last_stress, "time"])
                break
        break

# Monte Carlo tachycardia risk (simulation mode only — skip for CSV)
mc_prob: float | None = None
if data_mode != "📂 BioGears CSV Upload":
    rng_mc    = np.random.default_rng(99)
    risk_hits = 0
    for _ in range(mc_runs):
        sim_hr = df["heart_rate"] + rng_mc.normal(0, 3, len(df))
        if (sim_hr > 100).sum() > 10:
            risk_hits += 1
    mc_prob = risk_hits / mc_runs

# ===========================================================================
# DASHBOARD LAYOUT
# ===========================================================================

# ── Source badge ─────────────────────────────────────────────────────────────
st.caption(f"**Active data source:** {source_label}")

# ── Risk metric cards ────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Risk Metrics</div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)

# Helper to colour metric value
def _delta_colour(value, threshold, label_ok="Normal", label_bad="⚠️ Elevated"):
    return label_ok if value < threshold else label_bad

c1.metric(
    "Hypotension Events",
    hypo_events,
    delta=_delta_colour(hypo_events, 1, "✅ None", "⚠️ Detected"),
    delta_color="normal" if hypo_events == 0 else "inverse",
    help="Time steps where Mean Arterial Pressure < 65 mmHg (clinically critical threshold).",
)
c2.metric(
    "Tachycardia Events",
    tachy_events,
    delta=_delta_colour(tachy_events, 1, "✅ None", "⚠️ Detected"),
    delta_color="normal" if tachy_events == 0 else "inverse",
    help="Time steps where Heart Rate > 100 bpm.",
)
c3.metric(
    "Instability Index",
    f"{cii:.3f}",
    delta="Stable" if cii < 0.15 else ("Moderate" if cii < 0.25 else "⚠️ High"),
    delta_color="normal" if cii < 0.15 else ("off" if cii < 0.25 else "inverse"),
    help="(std(HR)/mean(HR)) + (std(MAP)/mean(MAP)). Higher = more variable cardiovascular state.",
)
c4.metric(
    "Recovery Time",
    "Not recovered" if recovery_time is None else f"{int(recovery_time)} min",
    delta="✅ OK" if recovery_time and recovery_time < 30 else ("—" if recovery_time is None else "⚠️ Slow"),
    delta_color="normal" if (recovery_time and recovery_time < 30) else "off",
    help="Minutes after re-entry for HR and MAP to return within 10 % of baseline.",
)
c5.metric(
    "Mean CV Load Index",
    f"{df['load_index'].mean():.0f}",
    delta="HR × CO",
    delta_color="off",
    help="Heart Rate × Cardiac Output — proxy for overall cardiac workload.",
)

# ── Status banners ───────────────────────────────────────────────────────────
if cii < 0.15:
    st.success("✅ Cardiovascular system is **stable** throughout the mission.")
elif cii < 0.25:
    st.warning("⚠️ **Moderate** cardiovascular instability detected. Monitor closely.")
else:
    st.error("🚨 **High** cardiovascular instability — consider mission adjustments.")

if hypo_events > 5:
    st.error(f"🩸 Frequent hypotension detected ({hypo_events} events). Risk of syncope.")
if tachy_events > 10:
    st.warning(f"💓 Significant tachycardia detected ({tachy_events} events).")

st.divider()

# ===========================================================================
# CHARTS
# ===========================================================================

# Phase-band colour map
PHASE_COLORS = {
    "Launch":         "rgba(231, 76, 60, 0.10)",
    "Microgravity":   "rgba(52, 152, 219, 0.10)",
    "Exercise":       "rgba(46, 204, 113, 0.10)",
    "Re-entry":       "rgba(230, 126, 34, 0.12)",
    "Recovery":       "rgba(149, 165, 166, 0.10)",
    # BioGears event names
    "baseline":               "rgba(52, 152, 219, 0.08)",
    "microgravity_fluid_shift":"rgba(52, 152, 219, 0.08)",
    "exercise_workload":       "rgba(46, 204, 113, 0.10)",
    "reentry_stress":          "rgba(230, 126, 34, 0.12)",
    "recovery":               "rgba(149, 165, 166, 0.10)",
}


def add_phase_bands(fig: go.Figure, df: pd.DataFrame) -> go.Figure:
    """Add transparent colour bands for each mission phase."""
    changes     = df["phase"].ne(df["phase"].shift())
    starts      = df.index[changes].tolist()
    starts.append(len(df))
    added_phases: set[str] = set()

    for i in range(len(starts) - 1):
        s, e   = starts[i], starts[i + 1] - 1
        phase  = df.iloc[s]["phase"]
        t0, t1 = df.iloc[s]["time"], df.iloc[e]["time"]
        color  = PHASE_COLORS.get(phase, "rgba(120,120,120,0.07)")
        show_label = phase not in added_phases
        fig.add_vrect(
            x0=t0,
            x1=t1,
            fillcolor=color,
            opacity=1,
            layer="below",
            line_width=0,
            annotation_text=phase if show_label else "",
            annotation_position="top left",
            annotation_font_size=9,
        )
        added_phases.add(phase)
    return fig


COMMON_LAYOUT = dict(
    paper_bgcolor="#0b1622",
    plot_bgcolor="#0d1b2a",
    font=dict(color="#94b8cc", family="Inter"),
    xaxis=dict(gridcolor="#1e3a5f", title="Mission Time (min)", zerolinecolor="#1e3a5f"),
    yaxis=dict(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f"),
    margin=dict(l=60, r=20, t=50, b=50),
    hovermode="x unified",
    legend=dict(bgcolor="#0b1622", bordercolor="#1e3a5f", borderwidth=1),
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["❤️ Heart Rate", "🩺 Mean Arterial Pressure", "⚡ CV Load Index", "🌌 Phase Space", "🔋 Cumulative Fatigue"])

# ── Heart Rate ───────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="graph-caption">Shows how heart rate changes across mission phases. Spikes during Exercise and Re-entry are expected. Sustained HR > 100 bpm is tachycardia, which is a physiological risk marker.</div>', unsafe_allow_html=True)
    fig_hr = go.Figure()
    fig_hr.add_trace(go.Scatter(
        x=df["time"], y=df["heart_rate"],
        mode="lines", name="Heart Rate",
        line=dict(color="#e74c3c", width=1.5),
        fill="tozeroy", fillcolor="rgba(231,76,60,0.06)",
    ))
    fig_hr.add_hline(y=100, line_dash="dash", line_color="#e74c3c", opacity=0.5,
                     annotation_text="Tachycardia threshold (100 bpm)", annotation_font_size=9)
    fig_hr = add_phase_bands(fig_hr, df)
    fig_hr.update_layout(
        title="Heart Rate vs Mission Time",
        yaxis_title="Heart Rate (bpm)",
        **COMMON_LAYOUT,
    )
    st.plotly_chart(fig_hr, use_container_width=True)

# ── MAP ──────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="graph-caption">Mean Arterial Pressure (MAP) reflects perfusion pressure to vital organs. Drops below 65 mmHg indicate hypotension — a critical threshold for organ perfusion in space medicine.</div>', unsafe_allow_html=True)
    fig_map = go.Figure()
    fig_map.add_trace(go.Scatter(
        x=df["time"], y=df["map"],
        mode="lines", name="MAP",
        line=dict(color="#3498db", width=1.5),
        fill="tozeroy", fillcolor="rgba(52,152,219,0.06)",
    ))
    fig_map.add_hline(y=65, line_dash="dash", line_color="#e74c3c", opacity=0.5,
                      annotation_text="Hypotension threshold (65 mmHg)", annotation_font_size=9)
    fig_map = add_phase_bands(fig_map, df)
    fig_map.update_layout(
        title="Mean Arterial Pressure vs Mission Time",
        yaxis_title="MAP (mmHg)",
        **COMMON_LAYOUT,
    )
    st.plotly_chart(fig_map, use_container_width=True)

# ── CV Load Index ─────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="graph-caption">The Cardiovascular Load Index (HR × Cardiac Output) is a proxy for total cardiac workload. Higher values mean the heart is working harder. Sustained high values indicate cardiovascular strain.</div>', unsafe_allow_html=True)
    fig_load = go.Figure()
    fig_load.add_trace(go.Scatter(
        x=df["time"], y=df["load_index"],
        mode="lines", name="CV Load Index",
        line=dict(color="#f39c12", width=1.5),
        fill="tozeroy", fillcolor="rgba(243,156,18,0.06)",
    ))
    fig_load = add_phase_bands(fig_load, df)
    fig_load.update_layout(
        title="Cardiovascular Load Index (HR × Cardiac Output)",
        yaxis_title="HR × CO",
        **COMMON_LAYOUT,
    )
    st.plotly_chart(fig_load, use_container_width=True)

# ── Phase Space ───────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="graph-caption"><strong>Dynamical Systems Phase Space:</strong> Plots Heart Rate against Mean Arterial Pressure. Evaluates cardiovascular stability by highlighting physiological "attractor states" across different mission phases. A dispersed or drifting scatter loop indicates degraded homeostatic control.</div>', unsafe_allow_html=True)
    fig_phase = go.Figure()
    
    # Predefined appealing color palette for phases
    for phase_name, group in df.groupby("phase"):
        phase_str = str(phase_name)
        color = PHASE_COLORS.get(phase_str, "rgba(200, 200, 200, 0.4)")
        # Replace alpha value with 1.0 for scatter markers
        marker_color = color.replace("0.10", "0.9").replace("0.12", "0.9").replace("0.08", "0.9")
        
        fig_phase.add_trace(go.Scatter(
            x=group["map"], y=group["heart_rate"],
            mode="markers+lines", name=phase_str,
            marker=dict(size=7, color=marker_color, line=dict(width=1, color="#ffffff")),
            line=dict(width=1, color=marker_color, dash="dot"),
        ))
    fig_phase.update_layout(
        title="Cardiovascular Phase Space Trajectory (MAP vs HR)",
        xaxis_title="Mean Arterial Pressure (mmHg)",
        yaxis_title="Heart Rate (bpm)",
        height=550,
        **COMMON_LAYOUT
    )
    # Remove x unified hovermode for scatter to show closest point
    fig_phase.update_layout(hovermode="closest")
    st.plotly_chart(fig_phase, use_container_width=True)

# ── Cumulative Fatigue ────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="graph-caption"><strong>Cumulative Systemic Fatigue:</strong> A stateful accumulator tracking overall physiological stress exposure over time. High values (>80) indicate severe exhaustion, reducing pilot adaptability and requiring scheduled rest.</div>', unsafe_allow_html=True)
    fig_fatigue = go.Figure()
    fig_fatigue.add_trace(go.Scatter(
        x=df["time"], y=df["fatigue_index"],
        mode="lines", name="Fatigue Index",
        line=dict(color="#9b59b6", width=2),
        fill="tozeroy", fillcolor="rgba(155, 89, 182, 0.15)",
    ))
    fig_fatigue.add_hline(y=80, line_dash="dash", line_color="#e74c3c", opacity=0.8,
                          annotation_text="Critical Exhaustion Threshold", annotation_font_size=10, annotation_position="top right")
    fig_fatigue = add_phase_bands(fig_fatigue, df)
    fig_fatigue.update_layout(
        title="Mission Cumulative Fatigue Tracking",
        yaxis_title="Fatigue Index (0-100)",
        yaxis_range=[0, 105],
        **COMMON_LAYOUT,
    )
    st.plotly_chart(fig_fatigue, use_container_width=True)

st.divider()

# ── Monte Carlo + Data table ──────────────────────────────────────────────────
st.markdown('<div class="section-title">Risk Forecast & Raw Data</div>', unsafe_allow_html=True)

col_mc, col_data = st.columns([1, 2])

with col_mc:
    st.subheader("📊 Monte Carlo Risk Forecast")
    if mc_prob is not None:
        colour = "🟢" if mc_prob < 0.2 else ("🟡" if mc_prob < 0.5 else "🔴")
        st.metric(
            "Tachycardia Risk Probability",
            f"{colour}  {round(mc_prob * 100, 1)} %",
            help="Fraction of Monte Carlo runs where HR exceeded 100 bpm for > 10 consecutive time steps.",
        )
        st.caption(f"Based on {mc_runs} simulated mission variations.")
        st.markdown(
            """
            **How it works:** The simulation runs many alternate versions of
            the same mission (each with small random physiological variations)
            and counts how often tachycardia occurs. A high probability
            signals that the astronaut is near a risk threshold.
            """
        )
    else:
        st.info("Monte Carlo is only available in Simulation Mode.")

with col_data:
    st.subheader("📋 Mission Event Log & Export")
    
    # Mission Event Log UI
    log_html = "<div style='height: 220px; overflow-y: auto; background-color: #0a1628; padding: 14px; border-radius: 8px; border: 1px solid #1e3a5f; margin-bottom: 16px; font-size: 0.85rem; color: #b0cfe8; box-shadow: inset 0 2px 10px rgba(0,0,0,0.3); font-family: monospace;'>"
    
    prev_phase = None
    event_count = 0
    for idx, row in df.iterrows():
        t = int(row["time"])
        cp = row["phase"]
        
        if cp != prev_phase and prev_phase is not None:
            log_html += f"<div style='margin-bottom: 8px; color: #4fc3f7;'>[T+{t:03d}] 🔄 TRANSITION: <i>{cp.upper()}</i> phase initiated.</div>"
            event_count += 1
        prev_phase = cp
        
        # Debounce alerts (report roughly once per segment if active)
        if row["tachycardia"] and (idx % 15 == 0 or idx == 0):
            log_html += f"<div style='margin-bottom: 8px; color: #f4a240;'>[T+{t:03d}] ⚠️ ALERT: Tachycardia detected (HR: {int(row['heart_rate'])} bpm).</div>"
            event_count += 1
            
        if row["hypotension"] and (idx % 15 == 0 or idx == 0):
            log_html += f"<div style='margin-bottom: 8px; color: #e74c3c;'>[T+{t:03d}] 🚨 CRITICAL: Hypotension condition (MAP: {int(row['map'])} mmHg).</div>"
            event_count += 1
            
        if row["fatigue_index"] > 80 and (idx % 30 == 0):
            log_html += f"<div style='margin-bottom: 8px; color: #9b59b6;'>[T+{t:03d}] 🛑 WARNING: Pilot fatigue critically high (Index: {row['fatigue_index']:.1f}).</div>"
            event_count += 1

    if event_count == 0:
         log_html += "<div style='margin-bottom: 8px; color: #2ecc71;'>[T+000] ✓ Mission telemetry nominally stable. No anomalies recorded.</div>"
         
    log_html += "</div>"
    st.markdown(log_html, unsafe_allow_html=True)

    # Export Button
    display_cols = [c for c in ["time", "phase", "heart_rate", "map", "cardiac_output",
                                "fatigue_index", "hypotension", "tachycardia", "load_index"] if c in df.columns]
    export_df = df[display_cols].round(3)
    csv_data = export_df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Download Analyzed Mission Telemetry (CSV)",
        data=csv_data,
        file_name='mission_telemetry_analyzed.csv',
        mime='text/csv',
        use_container_width=True,
        help="Export the analyzed dataset containing physiological indices and detection flags for external reporting."
    )

st.divider()
st.caption(
    "🛸 **ISRO Gaganyaan Cardiovascular Digital Twin** — BioGears-inspired |"
    " Space Medicine Division | Analytics layer accepts real BioGears CSV data"
)