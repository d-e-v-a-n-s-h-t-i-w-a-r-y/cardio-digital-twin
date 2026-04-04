"""
Cardiovascular Digital Twin — Full Pipeline
============================================
Step-by-step script that:
  1. Loads real BioGears CSV data  (biogears_real_data.csv at project root)
  2. Extends it with synthetic physiological data up to 48 hours
  3. Defines mission phases
  4. Generates sleep quality and fatigue index
  5. Saves the merged dataset to final_digital_twin_dataset.csv
  6. Plots Heart Rate, Sleep Quality, and Fatigue Index

BioGears CSV format expected
    Time(s)  |  CTSresistance  |  HeartRate(1/min)  |  RespirationRate(1/min)  |  MeanArterialPressure(mmHg)

Run from the project root:
    python digital_twin_full_pipeline.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless-safe backend (saves PNGs without needing a display)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# PATHS — resolved relative to this script so they work regardless of where
# you call the script from.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
REAL_DATA_CSV = ROOT / "biogears_real_data.csv"
OUTPUT_CSV    = ROOT / "final_digital_twin_dataset.csv"
PLOTS_DIR     = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# STEP 1: LOAD REAL BIOGEARS DATA
# ---------------------------------------------------------------------------
print("📂 Loading BioGears real data …")

if not REAL_DATA_CSV.exists():
    raise FileNotFoundError(
        f"Could not find: {REAL_DATA_CSV}\n"
        "Make sure biogears_real_data.csv is in the project root folder."
    )

real_df = pd.read_csv(REAL_DATA_CSV)

# Validate required columns
REQUIRED = {"Time(s)", "HeartRate(1/min)", "RespirationRate(1/min)", "MeanArterialPressure(mmHg)"}
missing_cols = REQUIRED - set(real_df.columns)
if missing_cols:
    raise ValueError(
        f"biogears_real_data.csv is missing columns: {missing_cols}\n"
        f"Found columns: {list(real_df.columns)}"
    )

# Convert time from seconds → hours
real_df["Time_hr"] = real_df["Time(s)"] / 3600.0

# Rename columns for clarity
real_df = real_df.rename(columns={
    "HeartRate(1/min)":          "HeartRate",
    "RespirationRate(1/min)":    "RespirationRate",
    "MeanArterialPressure(mmHg)":"MAP",
})

# Keep only the columns we need
real_df = real_df[["Time_hr", "HeartRate", "RespirationRate", "MAP"]].copy()

print(f"   ✅ Loaded {len(real_df)} rows — time range: "
      f"{real_df['Time_hr'].min():.2f} h → {real_df['Time_hr'].max():.2f} h")

# ---------------------------------------------------------------------------
# STEP 2: CREATE SYNTHETIC TIME (last real timestamp → 48 hours)
# ---------------------------------------------------------------------------
last_real_time = real_df["Time_hr"].max()
TARGET_HOURS   = 48.0
STEP_HR        = 1 / 60.0          # one row per minute

if last_real_time >= TARGET_HOURS:
    print(f"   ℹ️  Real data already covers {last_real_time:.1f} h — no synthetic extension needed.")
    fake_df = pd.DataFrame(columns=["Time_hr", "HeartRate", "RespirationRate", "MAP"])
else:
    fake_time = np.arange(last_real_time + STEP_HR, TARGET_HOURS, STEP_HR)
    fake_df = pd.DataFrame({"Time_hr": fake_time})
    print(f"   🔧 Generating {len(fake_df)} synthetic rows ({last_real_time:.2f} h → {TARGET_HOURS} h)")

# ---------------------------------------------------------------------------
# STEP 3: GENERATE SYNTHETIC PHYSIOLOGICAL DATA
# ---------------------------------------------------------------------------
rng = np.random.default_rng(seed=42)

if not fake_df.empty:
    fake_df["HeartRate"]       = rng.normal(72,  3, len(fake_df))
    fake_df["RespirationRate"] = rng.normal(14,  1, len(fake_df))
    fake_df["MAP"]             = rng.normal(78,  2, len(fake_df))

# ---------------------------------------------------------------------------
# STEP 4: DEFINE MISSION / SCENARIO PHASES
# ---------------------------------------------------------------------------
def get_phase(t_hr: float) -> str:
    """Map elapsed mission time (hours) to a phase label."""
    if t_hr < 6:
        return "Normal"
    elif t_hr < 18:
        return "MotionSickness"
    elif t_hr < 30:
        return "SleepDisruption"
    else:
        return "Recovery"

real_df["Phase"] = real_df["Time_hr"].apply(get_phase)
if not fake_df.empty:
    fake_df["Phase"] = fake_df["Time_hr"].apply(get_phase)

# ---------------------------------------------------------------------------
# STEP 5: SLEEP QUALITY (phase-derived)
# ---------------------------------------------------------------------------
SLEEP_MAP = {
    "Normal":          80,
    "MotionSickness":  60,
    "SleepDisruption": 40,
    "Recovery":        70,
}

real_df["SleepQuality"] = real_df["Phase"].map(SLEEP_MAP)
if not fake_df.empty:
    fake_df["SleepQuality"] = fake_df["Phase"].map(SLEEP_MAP)

# ---------------------------------------------------------------------------
# STEP 6: FATIGUE INDEX (stateful accumulation across merged timeline)
# ---------------------------------------------------------------------------
combined_phases = list(real_df["Phase"]) + (list(fake_df["Phase"]) if not fake_df.empty else [])

fatigue_vals: list[float] = []
current_fatigue = 20.0          # starting fatigue level (0–100 scale)

FATIGUE_DELTA = {
    "MotionSickness":  +8.0,
    "SleepDisruption": +6.0,
    "Recovery":        -4.0,
    "Normal":          +1.0,
}

for phase in combined_phases:
    current_fatigue += FATIGUE_DELTA.get(phase, 0.0)
    current_fatigue = float(np.clip(current_fatigue, 0.0, 100.0))
    fatigue_vals.append(current_fatigue)

# ---------------------------------------------------------------------------
# STEP 7: MERGE REAL + SYNTHETIC DATA
# ---------------------------------------------------------------------------
final_df = pd.concat([real_df, fake_df], ignore_index=True)
final_df["FatigueIndex"] = fatigue_vals
final_df["RiskFlag"]     = final_df["FatigueIndex"] > 80

# ---------------------------------------------------------------------------
# STEP 8: SAVE FINAL DATASET
# ---------------------------------------------------------------------------
final_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Final dataset saved → {OUTPUT_CSV}  ({len(final_df)} rows)")
print(final_df[["Time_hr", "HeartRate", "MAP", "Phase", "FatigueIndex", "RiskFlag"]].describe())

# ---------------------------------------------------------------------------
# STEP 9: PLOTS (saved as PNG files — safe for headless / VS Code terminal)
# ---------------------------------------------------------------------------
PHASE_COLORS = {
    "Normal":          "#4C9BE8",
    "MotionSickness":  "#E87C4C",
    "SleepDisruption": "#9B59B6",
    "Recovery":        "#2ECC71",
}

def add_phase_shading(ax, df: pd.DataFrame):
    """Add colored vertical bands for each mission phase."""
    changes = df["Phase"].ne(df["Phase"].shift())
    starts  = df.index[changes].tolist()
    starts.append(len(df))
    for i in range(len(starts) - 1):
        s, e   = starts[i], starts[i + 1] - 1
        phase  = df.iloc[s]["Phase"]
        t0, t1 = df.iloc[s]["Time_hr"], df.iloc[e]["Time_hr"]
        ax.axvspan(t0, t1, alpha=0.12, color=PHASE_COLORS.get(phase, "#888888"), label=phase)

# -- Heart Rate --
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(final_df["Time_hr"], final_df["HeartRate"], color="#E74C3C", linewidth=0.8)
add_phase_shading(ax, final_df)
ax.set_xlabel("Mission Time (hours)")
ax.set_ylabel("Heart Rate (bpm)")
ax.set_title("Heart Rate vs Mission Time")
ax.axhline(100, color="red", linestyle="--", linewidth=0.8, label="Tachycardia threshold (100 bpm)")
# Deduplicate legend entries
handles, labels = ax.get_legend_handles_labels()
seen = {}
for h, l in zip(handles, labels):
    seen.setdefault(l, h)
ax.legend(list(seen.values()), list(seen.keys()), fontsize=7, loc="upper right")
plt.tight_layout()
fig.savefig(PLOTS_DIR / "heart_rate.png", dpi=150)
plt.close(fig)
print(f"📊 Saved: {PLOTS_DIR / 'heart_rate.png'}")

# -- Sleep Quality --
fig, ax = plt.subplots(figsize=(14, 3))
ax.plot(final_df["Time_hr"], final_df["SleepQuality"], color="#3498DB", linewidth=0.9)
add_phase_shading(ax, final_df)
ax.set_xlabel("Mission Time (hours)")
ax.set_ylabel("Sleep Quality (0–100)")
ax.set_title("Sleep Quality vs Mission Time")
plt.tight_layout()
fig.savefig(PLOTS_DIR / "sleep_quality.png", dpi=150)
plt.close(fig)
print(f"📊 Saved: {PLOTS_DIR / 'sleep_quality.png'}")

# -- Fatigue Index --
fig, ax = plt.subplots(figsize=(14, 3))
ax.plot(final_df["Time_hr"], final_df["FatigueIndex"], color="#F39C12", linewidth=0.9)
add_phase_shading(ax, final_df)
ax.axhline(80, color="red", linestyle="--", linewidth=0.8, label="Risk threshold (80)")
ax.set_xlabel("Mission Time (hours)")
ax.set_ylabel("Fatigue Index (0–100)")
ax.set_title("Fatigue Index vs Mission Time")
ax.fill_between(final_df["Time_hr"], 80, final_df["FatigueIndex"],
                where=final_df["FatigueIndex"] > 80,
                alpha=0.25, color="red", label="High-risk zone")
ax.legend(fontsize=8, loc="upper left")
plt.tight_layout()
fig.savefig(PLOTS_DIR / "fatigue_index.png", dpi=150)
plt.close(fig)
print(f"📊 Saved: {PLOTS_DIR / 'fatigue_index.png'}")

print("\n🚀 Pipeline complete.")
