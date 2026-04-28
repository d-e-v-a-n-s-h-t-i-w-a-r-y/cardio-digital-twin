# 🛸 Astronaut Cardiovascular Digital Twin
### ISRO · Gaganyaan Human Spaceflight Programme · Space Medicine Division

> **A BioGears-inspired real-time cardiovascular risk monitoring system for astronauts across space mission phases — featuring live simulation, physiological analytics, Monte Carlo risk forecasting, and an interactive Streamlit dashboard.**

---

## 📌 Table of Contents
1. [What is a Digital Twin?](#what-is-a-digital-twin)
2. [Project Overview](#project-overview)
3. [System Architecture](#system-architecture)
4. [Data Sources](#data-sources)
5. [Mission Phases](#mission-phases)
6. [Physiological Metrics & Formulas](#physiological-metrics--formulas)
7. [Analytics Modules (with Formulas)](#analytics-modules-with-formulas)
8. [Simulation Engine](#simulation-engine)
9. [Dashboard Features](#dashboard-features)
10. [Monte Carlo Risk Forecasting](#monte-carlo-risk-forecasting)
11. [Tech Stack](#tech-stack)
12. [Project Structure](#project-structure)
13. [How to Run](#how-to-run)
14. [Glossary of Medical Terms](#glossary-of-medical-terms)

---

## 🧬 What is a Digital Twin?

A **Digital Twin** is a virtual, real-time replica of a physical system — in this case, the **cardiovascular system of an astronaut**. Instead of continuously testing on a human, sensors or simulation models feed data into a computational model that behaves like the real system. Doctors and mission controllers can then:

- **Monitor** physiological health in real-time
- **Simulate** "what-if" scenarios (e.g., *what happens to heart rate during re-entry?*)
- **Predict** risks before they become emergencies
- **Recommend** countermeasures (rest, fluid intake, medication)

> In space medicine, this is critical because once a spacecraft is launched, doctors cannot physically intervene. The digital twin acts as an early warning system.

---

## 🚀 Project Overview

| Property | Details |
|---|---|
| **Domain** | Space Medicine / Biomedical Engineering |
| **Mission** | ISRO Gaganyaan (India's first crewed spaceflight) |
| **Purpose** | Monitor, simulate, and forecast astronaut cardiovascular health |
| **Key Risk Events** | Tachycardia, Hypotension, High Fatigue, Cardiovascular Instability |
| **Data Source** | BioGears open-source physiology engine data + synthetic simulation |
| **Interface** | Interactive Streamlit web dashboard |
| **Operating Modes** | (1) Simulation Mode — synthetic physiology from user inputs, (2) BioGears CSV Upload — real physiological data |

### Key Capabilities
- ✅ Real-time trace plots of Heart Rate, MAP, CV Load Index, Phase Space, and Fatigue
- ✅ Risk detection with clinical thresholds (Hypotension < 65 mmHg, Tachycardia > 100 bpm)
- ✅ Cardiovascular Instability Index computation
- ✅ Recovery time estimation after re-entry stress
- ✅ Monte Carlo probabilistic risk forecasting
- ✅ Mission Event Log with automated alerts
- ✅ CSV export for external medical reporting

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   CARDIOVASCULAR DIGITAL TWIN                   │
├──────────────────┬──────────────────┬───────────────────────────┤
│   DATA LAYER     │  SIMULATION LAYER │     ANALYTICS LAYER       │
│                  │                  │                           │
│ BioGears CSV     │ Event Simulator  │  Risk Detection           │
│ (Real data)      │ (Phase-by-phase  │  (Hypotension/Tachy)      │
│     ↓            │  physio model)   │        ↓                  │
│ Synthetic Data   │       ↓          │  Instability Index        │
│ (48-hr extension)│ Probabilistic    │  (CII formula)            │
│     ↓            │ Inputs           │        ↓                  │
│ Pipeline Script  │ (Normal/Poisson/ │  Recovery Estimator       │
│ (Merge + Fatigue)│  Lognormal)      │  (10% tolerance rule)     │
│                  │                  │        ↓                  │
│                  │                  │  Monte Carlo Engine       │
│                  │                  │  (N=50–500 runs)          │
├──────────────────┴──────────────────┴───────────────────────────┤
│                     VISUALIZATION LAYER                         │
│        Streamlit Dashboard — 5 interactive chart tabs           │
│    Heart Rate | MAP | CV Load | Phase Space | Fatigue           │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
biogears_real_data.csv
        ↓
  Load & validate columns (Time, HR, RespirationRate, MAP)
        ↓
  Extend to 48 hours with synthetic Gaussian noise data
        ↓
  Assign Mission Phases (Normal → MotionSickness → SleepDisruption → Recovery)
        ↓
  Compute Sleep Quality score per phase
        ↓
  Compute Cumulative Fatigue Index (stateful accumulator)
        ↓
  Merge real + synthetic → final_digital_twin_dataset.csv
        ↓
  Dashboard reads CSV or runs real-time simulation
        ↓
  Analytics → Risk Detection → Monte Carlo → Display
```

---

## 📂 Data Sources

### 1. BioGears Engine Data (`biogears_real_data.csv`)
**BioGears** is an open-source, high-fidelity, whole-body physiology simulation engine developed by TechAnalytic Solutions (funded by the US Army). It models organ-level physiology including:
- Cardiovascular system (heart, vasculature)
- Respiratory system
- Renal system
- Sympathetic/Parasympathetic nervous system responses

**Columns expected:**
| Column | Unit | Description |
|---|---|---|
| `Time(s)` | seconds | Elapsed simulation time |
| `HeartRate(1/min)` | beats/min (bpm) | Heart rate |
| `RespirationRate(1/min)` | breaths/min | Breathing rate |
| `MeanArterialPressure(mmHg)` | mmHg | Average blood pressure |

### 2. Synthetic Extension Data
When real data covers < 48 hours, the pipeline extends the dataset using **Gaussian (Normal) distribution** sampling:

```python
HeartRate       ~ N(μ=72,  σ=3)  bpm
RespirationRate ~ N(μ=14,  σ=1)  breaths/min
MAP             ~ N(μ=78,  σ=2)  mmHg
```

Where:
- **μ (mu)** = mean (average expected value)
- **σ (sigma)** = standard deviation (spread of variation)
- **N(μ, σ)** = Normal (Gaussian) distribution

> The seed `42` is fixed for reproducibility — every run generates the same synthetic values.

---

## 🌍 Mission Phases

The system divides a mission into distinct **phases** based on elapsed time. Each phase represents a known medical challenge in spaceflight:

### Pipeline Phases (48-hour dataset)
| Phase | Time Range | Medical Context |
|---|---|---|
| **Normal** | 0 – 6 hours | Baseline pre-launch physiology |
| **Motion Sickness** | 6 – 18 hours | Space motion sickness from vestibular system disruption; elevated HR, nausea |
| **Sleep Disruption** | 18 – 30 hours | Circadian rhythm desynchronization; reduced sleep quality → fatigue accumulation |
| **Recovery** | 30 – 48 hours | Body adapts; fatigue decreases, vitals stabilize |

### Dashboard Simulation Phases (260-min default)
| Phase | Time Range | Medical Context |
|---|---|---|
| **Launch** | 0 – 40 min | G-force stress; HR and MAP spike due to sympathetic nervous system activation |
| **Microgravity** | 40 – 120 min | Fluid shifts headward (cephalad fluid shift); MAP stable, HR elevated initially, CO rises |
| **Exercise** | 120 – 200 min | Scheduled in-flight exercise protocol; HR rises proportional to intensity |
| **Re-entry** | 200 – 230 min | Deceleration G-forces (+Gx); HR spikes, MAP remains stable or rises due to reclined seat |
| **Recovery** | 230+ min | Return to Earth; orthostatic deconditioning; vitals gradually return to baseline |

---

## 📏 Physiological Metrics & Formulas

### 1. Heart Rate (HR)
**Definition:** Number of heartbeats per minute.  
**Unit:** bpm (beats per minute)  
**Clinical thresholds:**
- Normal: 60 – 100 bpm
- **Tachycardia (risk):** > 100 bpm
- Bradycardia: < 60 bpm

**In simulation (example — Exercise phase):**
```
HR_exercise = HR_resting + 50 × exercise_intensity + N(0, 3σ_noise)
```
Where `exercise_intensity ∈ [0, 1]` — a user-controlled slider.

---

### 2. Mean Arterial Pressure (MAP)
**Definition:** The average blood pressure in the arteries during one cardiac cycle. It represents the perfusion pressure that drives blood (and therefore oxygen) to vital organs.

**Clinical Formula:**
```
MAP = DBP + (1/3) × (SBP - DBP)
```
Or equivalently:
```
MAP = (SBP + 2 × DBP) / 3
```
Where:
- **SBP** = Systolic Blood Pressure (pressure when heart contracts)
- **DBP** = Diastolic Blood Pressure (pressure when heart relaxes)

**Unit:** mmHg (millimetres of mercury)  
**Clinical thresholds:**
- Normal: 70 – 105 mmHg
- **Hypotension (risk):** < 65 mmHg ← *critical organ perfusion failure threshold*
- Hypertension: > 110 mmHg

> In space, microgravity causes **cephalad fluid shift** — blood moves from the legs toward the head/chest. While MAP remains relatively stable, the accompanying plasma volume loss (diuresis) causes **orthostatic intolerance** upon return to gravity.

---

### 3. Cardiac Output (CO)
**Definition:** Volume of blood the heart pumps per minute.

**Physiological Formula:**
```
CO = HR × SV
```
Where:
- **HR** = Heart Rate (bpm)
- **SV** = Stroke Volume (mL per beat) — volume ejected per heartbeat

**Unit:** L/min (litres per minute)  
**Normal range:** 4 – 8 L/min (average ~5.2 L/min at rest)

| Profile | Resting CO |
|---|---|
| Elite Athlete | 6.5 L/min |
| Average Astronaut | 5.2 L/min |
| Deconditioned | 4.0 L/min |

In simulation, CO is modulated with a sinusoidal wave during exercise:
```
CO(t) = CO_baseline + exercise_intensity × sin(t / 20) × 0.5
```

---

### 4. Systolic & Diastolic Blood Pressure (SBP / DBP)
During simulation, these are derived from MAP with Gaussian noise:
```
SBP = MAP + 18 + N(0, 1.8)   [mmHg]
DBP = MAP - 10 + N(0, 1.2)   [mmHg]
```

---

### 5. Sleep Quality Index
A **phase-derived ordinal score** (0–100 scale) representing estimated sleep quality:

| Phase | Sleep Quality Score |
|---|---|
| Normal | 80 |
| Motion Sickness | 60 |
| Sleep Disruption | 40 |
| Recovery | 70 |

> Sleep quality directly feeds into astronaut cognitive performance and fatigue accumulation. Disrupted sleep in space is primarily caused by microgravity, noise, CO₂ levels, and schedule misalignment.

---

### 6. Respiration Rate
**Definition:** Number of breaths per minute.  
**Unit:** breaths/min  
**Normal range:** 12 – 20 breaths/min  
Synthetic values sampled from: `RespirationRate ~ N(14, 1)`

---

## 🧮 Analytics Modules (with Formulas)

### Module 1: Cardiovascular Instability Index (CII)
**File:** `analytics/instability.py`

The CII measures how variable (unstable) the cardiovascular system is throughout the mission. It uses the **Coefficient of Variation** — a normalized measure of statistical spread — applied to both HR and MAP.

**Formula:**
```
CII = (σ_HR / μ_HR) + (σ_MAP / μ_MAP)
```
Where:
- **σ** (sigma) = standard deviation of values across the mission
- **μ** (mu) = mean (average) of values across the mission

Both terms are **dimensionless** ratios (the units cancel out), so CII is a pure index.

**Interpretation:**
| CII Value | Interpretation | Action |
|---|---|---|
| < 0.15 | Stable ✅ | No action needed |
| 0.15 – 0.25 | Moderate instability ⚠️ | Close monitoring required |
| > 0.25 | High instability 🚨 | Consider mission adjustments |

**Why two terms?** We sum HR variability and MAP variability to get a holistic view. A system could have an unstable HR but stable MAP (or vice versa); only the sum captures the complete cardiovascular state.

---

### Module 2: Cardiovascular Load Index (CLI)
**File:** `analytics/load_index.py`

A proxy for **total cardiac workload** — how hard the heart is working at any given moment.

**Formula:**
```
CLI = HR × CO
```
Where:
- **HR** = Heart Rate (bpm)
- **CO** = Cardiac Output (L/min)

**Unit:** bpm · L/min (no standard name; used as a relative workload index)

**Example:**
- At rest: 70 bpm × 5.2 L/min = **364 units**
- During intense exercise: 150 bpm × 8.0 L/min = **1200 units**

High sustained values indicate the heart is under prolonged strain, increasing risk of cardiovascular fatigue.

---

### Module 3: Cumulative Fatigue Index
**Files:** `digital_twin_full_pipeline.py`, `visualization/dashboard.py`

A **stateful accumulator** model — it remembers previous states. Unlike instantaneous measurements, it tracks accumulated physiological stress over time, similar to how a runner accumulates muscle fatigue during a marathon.

**Algorithm:**
```
FatigueIndex[0] = 20.0   (initial starting value)

For each time step t:
    FatigueIndex[t] = clip(FatigueIndex[t-1] + Δ(phase[t]), 0, 100)
```

**Phase Deltas (Δ) — Pipeline version:**
| Phase | Δ (change per step) |
|---|---|
| Motion Sickness | +8.0 |
| Sleep Disruption | +6.0 |
| Normal | +1.0 |
| Recovery | −4.0 |

**Phase Deltas (Δ) — Dashboard Simulation version:**
| Phase | Δ (change per minute) |
|---|---|
| Microgravity / Motion | +0.15 |
| Exercise / Workload | +0.30 |
| Re-entry / Stress | +0.50 |
| Recovery | −0.10 |
| Other (baseline) | +0.02 |

**`clip(x, 0, 100)`** clamps the value between 0 and 100 so it never goes out of range.

**Threshold:** FatigueIndex > **80** → Critical exhaustion (🔴 High-risk zone)

> This model is analogous to the **ACWR (Acute:Chronic Workload Ratio)** used in sports science to predict injury risk.

---

### Module 4: Recovery Time Estimator
**File:** `analytics/recovery.py`

Measures how long it takes for the astronaut's vitals to return to baseline after the maximum-stress event (re-entry).

**Algorithm:**
1. **Baseline** = mean of HR and MAP from the first `n` rows before any stress (default `n=6` samples)
2. **Bounds computation:**
   ```
   HR_lower  = baseline_HR  × (1 − 0.10)
   HR_upper  = baseline_HR  × (1 + 0.10)
   MAP_lower = baseline_MAP × (1 − 0.10)
   MAP_upper = baseline_MAP × (1 + 0.10)
   ```
3. After the last re-entry row, scan forward. Count how many **consecutive** rows satisfy both:
   ```
   HR_lower ≤ HR[t] ≤ HR_upper   AND   MAP_lower ≤ MAP[t] ≤ MAP_upper
   ```
4. When **6 consecutive** such rows are found → the time elapsed since last re-entry = **Recovery Time**

**Interpretation:**
- Recovery Time < 30 min → ✅ Normal
- Recovery Time ≥ 30 min → ⚠️ Slow recovery (potential orthostatic intolerance)
- No recovery in dataset → "Not recovered" (severe concern)

> **Clinical relevance:** Prolonged recovery after G-force stress is a known marker of orthostatc intolerance — astronauts who cannot stand up after landing without fainting. ISRO needs to pre-screen Gaganyaan candidates for this.

---

### Module 5: Risk Detection (Binary Flags)
**File:** `analytics/risk_detection.py`

Simple threshold-based binary flagging:

| Flag | Condition | Clinical Significance |
|---|---|---|
| `hypotension_risk` | MAP < 65 mmHg | Organs may not receive enough blood; risk of syncope (fainting) |
| `tachycardia` | HR > 100 bpm | Heart working too fast; may indicate stress, dehydration, or arrhythmia |
| `instability_risk` | CII ≥ 0.15 | Volatile cardiovascular regulation |

**Composite Risk Score:**
```
risk_score[t] = hypotension_risk[t] + tachycardia[t] + instability_risk[t]
```
Values: 0 (no risk) to 3 (all three risks simultaneously present).

---

## ⚙️ Simulation Engine

### Event Simulator (`simulation/event_simulator.py`)
The heart of the simulation. Reads a baseline BioGears dataset and transforms it phase-by-phase with physiologically-plausible modifications:

**Microgravity Fluid Shift (t = 30–90 min):**
```
HR   = initial rise +3 to +5 bpm, adapting to near-baseline
MAP  = stable (±1.5 mmHg noise only)
CO   = rises by 18-41% due to increased stroke volume
PV   = Plasma Volume decays slowly (~12% loss over mission)
```
*Rationale:* In microgravity, ~2 litres of blood shift from the legs to the chest/head. The heart receives more blood (↑CO) and stroke volume increases, but MAP remains relatively stable.

**Exercise Workload (t = 90–150 min):**
```
intensity drawn from Gamma(k=2.0, θ=0.18)
HR   += (45 + 25 × intensity) × intensity
MAP  += (0.8 + 1.5 × intensity) × intensity
CO   += (2.0 + 2.5 × intensity) × intensity × sv_index
```
*Why Gamma distribution?* Exercise intensity in real populations is right-skewed and lower-bounded.

**Re-entry Stress (t = 150–180 min):**
```
Gx   = up to 4.0 G (Soyuz/Gaganyaan bell curve profile)
Gz_equiv = Gx × 0.55 [CV-equivalent Gz for 70° reclined seat]
HR   += 10 bpm per CV-equivalent Gz + onset rate bonus
MAP  += 2.8 mmHg per CV-equiv Gz (if < 3.5 G)
CO   = modulates via HR and Stroke Volume penalty
```

**Recovery (t > 180 min):**
```
Applies dual-exponential orthostatic readaptation:
Fast decay (τ = 18 min): autonomic cardiovascular reflex
Slow decay (τ = 280 min): plasma volume restoration
HR   += peak_hr_rise × (0.55 fast + 0.45 slow)
MAP  -= peak_map_drop × (0.45 fast + 0.55 slow)
CO   -= peak_co_drop × fast_decay × 0.6
```

### Probabilistic Inputs (`simulation/probabilistic_inputs.py`)
Three statistical distributions used to model physiological variability:

| Distribution | Formula | Used For |
|---|---|---|
| **Normal / Gaussian** | `X ~ N(μ, σ)` | Random physiologic noise on HR, MAP, CO |
| **Poisson** | `X ~ Poisson(λ = rate × duration)` | Count of random stress events during mission |
| **Gamma** | `X ~ Gamma(k, θ)` | Exercise intensity (right-skewed, lower-bounded) |

---

## 📊 Dashboard Features

### Mode 1: Simulation Mode
User controls the simulation via sidebar inputs:
- **Astronaut Profile** — Elite Athlete / Average Astronaut / Deconditioned (sets physiological baselines)
- **Resting HR** — base heart rate (bpm)
- **Baseline MAP** — base arterial pressure (mmHg)
- **Baseline Cardiac Output** — resting CO (L/min)
- **Exercise Intensity** — 0.0 (rest) to 1.0 (maximal effort)
- **Mission Duration** — 100 to 400 minutes
- **Physiologic Noise** — biological variability level
- **Monte Carlo Runs** — number of risk simulations (50–500)

### Mode 2: BioGears CSV Upload
Accepts any project CSV with **automatic column detection**. The system checks for multiple candidate column names (e.g., `HeartRate`, `heart_rate_bpm`, `HR`) and maps them to internal standard names. If cardiac output is missing, defaults to 5.2 L/min. Time units (seconds/hours/minutes) are auto-converted.

### 5 Chart Tabs

| Tab | Metric | Clinical Purpose |
|---|---|---|
| ❤️ Heart Rate | HR (bpm) over time | Detect tachycardia, stress response |
| 🩺 Mean Arterial Pressure | MAP (mmHg) over time | Detect hypotension, organ perfusion risk |
| ⚡ CV Load Index | HR × CO over time | Total cardiac workload / strain proxy |
| 🌌 Phase Space | HR vs MAP scatter | Cardiovascular attractor states; homeostasis assessment |
| 🔋 Cumulative Fatigue | Fatigue Index (0–100) over time | Systemic exhaustion tracking |

### Phase Space Plot (Tab 4 — Advanced)
This plot is inspired by **dynamical systems theory**. In a healthy, regulated system, all (MAP, HR) points should cluster tightly around a stable "attractor" (the body's homeostatic setpoint). A dispersed or drifting cloud of points indicates **degraded homeostatic control** — the body is struggling to regulate itself. Each mission phase is colour-coded.

### Mission Event Log
A real-time scrollable log that records:
- Phase transitions
- Tachycardia alerts (debounced every 15 time steps)
- Hypotension critical alerts
- High fatigue warnings (>80, logged every 30 steps)

---

## 🎲 Monte Carlo Risk Forecasting

**File:** `analytics/monte_carlo.py`

Monte Carlo simulation is a mathematical technique that runs an experiment **many times** with slightly different random inputs to understand the **probability distribution** of outcomes. Named after the Monte Carlo casino because of its reliance on randomness.

### How It Works Here

1. Take the **base mission profile** (HR, MAP, CO time series)
2. For each of `N` runs (default 200):
   - Add **Gaussian noise** to the entire HR, MAP, CO series:
     ```
     HR_run   = clip(HR_base   + N(0, σ_HR=2.0),   35, 190)
     MAP_run  = clip(MAP_base  + N(0, σ_MAP=1.5),  40, 130)
     CO_run   = clip(CO_base   + N(0, σ_CO=0.12),  2.0, 12.0)
     ```
   - Run full analytics (CII, risk detection, recovery) on the noisy version
   - Count whether tachycardia / hypotension / instability occurred
3. **Probability = (number of runs where event occurred) / N**

**Outputs:**
```
P(hypotension)  = hypotension_hits / N
P(tachycardia)  = tachycardia_hits / N
P(instability)  = instability_hits / N
E[recovery_time] = mean of all recovery times across runs
```

**Dashboard display (simplified version):**
```
P(tachycardia) = (runs where HR > 100 for > 10 steps) / mc_runs
```

**Colour-coded risk:**
- 🟢 < 20% probability → Low risk
- 🟡 20–50% → Moderate risk
- 🔴 > 50% → High risk

> **Why is this powerful?** A single simulation might show HR just barely under 100 bpm — looking safe. But if 70% of Monte Carlo variants push it over, the astronaut is actually at high risk. Monte Carlo reveals the risk that hides behind average values.

---

## 🧰 Tech Stack

| Library | Version | Purpose |
|---|---|---|
| **Python** | 3.11+ | Core programming language |
| **Pandas** | ≥ 2.0 | Data manipulation, DataFrame operations |
| **NumPy** | ≥ 1.24 | Numerical computation, statistical distributions |
| **Plotly** | ≥ 5.20 | Interactive, browser-rendered charts |
| **Streamlit** | ≥ 1.32 | Web dashboard framework |
| **Matplotlib** | Any | Static PNG plot generation in pipeline script |

### Why Streamlit?
Streamlit converts Python scripts into interactive web apps with zero HTML/CSS/JS expertise required. Widgets (sliders, dropdowns, file uploaders) are declared as Python one-liners. Ideal for rapid data science prototyping.

---

## 📁 Project Structure

```
cardio_digital_twin/
│
├── digital_twin_full_pipeline.py       # Standalone pipeline: load → extend → fatigue → plot → save CSV
├── main.py                             # Entry point (imports and runs pipeline)
├── requirements.txt                    # Python package dependencies
├── biogears_real_data.csv              # Raw BioGears engine output (real data)
├── final_digital_twin_dataset.csv      # Output: merged real + synthetic dataset (with fatigue)
│
├── simulation/                         # Physics / physiology simulation layer
│   ├── digital_twin_model.py           # High-level CardiovascularDigitalTwin class
│   ├── event_simulator.py              # Phase-by-phase physiological simulation
│   ├── probabilistic_inputs.py         # Statistical distributions (Normal, Poisson, LogNormal)
│   └── load_biogears_data.py           # BioGears CSV loader and validator
│
├── analytics/                          # Medical analytics and risk metrics
│   ├── instability.py                  # Cardiovascular Instability Index (CII)
│   ├── load_index.py                   # Cardiovascular Load Index (HR × CO)
│   ├── monte_carlo.py                  # Monte Carlo probabilistic risk simulation
│   ├── recovery.py                     # Post-stress recovery time estimator
│   └── risk_detection.py              # Binary risk flagging (hypotension, tachycardia)
│
├── visualization/                      # Dashboard and display layer
│   ├── dashboard.py                    # Main Streamlit app (752 lines)
│   ├── health_indicator.py             # Health status helper
│   └── timeline.py                     # Timeline chart helpers
│
├── plots/                              # Auto-generated PNG charts
│   ├── heart_rate.png
│   ├── sleep_quality.png
│   └── fatigue_index.png
│
└── .streamlit/
    └── config.toml                     # Streamlit theme configuration
```

---

## 🚀 How to Run

### Prerequisites
```bash
# Python 3.11+ required
pip install -r requirements.txt
```

### Step 1: Run the Data Pipeline
Generates the merged 48-hour dataset and static PNG charts:
```bash
python digital_twin_full_pipeline.py
```
**Output:** `final_digital_twin_dataset.csv` and `plots/*.png`

### Step 2: Launch the Interactive Dashboard
```bash
streamlit run visualization/dashboard.py
```
**Output:** Opens in browser at `http://localhost:8501`

### Using the Dashboard
1. **Simulation Mode** (default): Adjust sidebar sliders → charts update live
2. **CSV Upload Mode**: Click "📂 BioGears CSV Upload" → upload any project CSV → automatic column mapping → analytics run on real data

---

## 📖 Glossary of Medical Terms

| Term | Definition |
|---|---|
| **Tachycardia** | Heart rate > 100 bpm; can indicate physical or emotional stress, dehydration, or arrhythmia |
| **Bradycardia** | Heart rate < 60 bpm; common in athletes, but pathological in space context |
| **Hypotension** | Blood pressure too low (MAP < 65 mmHg); risks inadequate blood supply to brain and organs |
| **Hypertension** | Blood pressure too high (MAP > 110 mmHg); risk of stroke or cardiac event |
| **MAP** | Mean Arterial Pressure: average pressure driving blood to organs |
| **Cardiac Output (CO)** | Volume of blood pumped by heart per minute (L/min) |
| **Stroke Volume (SV)** | Blood ejected per heartbeat (mL/beat); CO = HR × SV |
| **Orthostatic Intolerance** | Inability to maintain blood pressure when standing / in gravity; a major post-flight problem |
| **Cephalad Fluid Shift** | Fluid moves from lower body toward head in microgravity; causes facial puffiness and MAP changes |
| **Homeostasis** | Body's ability to maintain stable internal conditions (temperature, pressure, pH, etc.) |
| **Attractor State** | In dynamical systems: a stable equilibrium the system naturally returns to after disturbance |
| **Coefficient of Variation** | σ/μ — normalized measure of dispersion; used in CII to compare variability across different scales |
| **Poisson Distribution** | Statistical model for counting random events over time (e.g., stress spikes during a mission) |
| **Log-Normal Distribution** | Right-skewed distribution; always positive; used for exercise intensity modelling |
| **Gaussian / Normal Distribution** | Bell-curve distribution; models most biological noise |
| **Monte Carlo Simulation** | Repeated random sampling to estimate probabilities of outcomes |
| **Syncope** | Fainting caused by brief loss of blood flow to the brain; risk during hypotension |
| **ACWR** | Acute:Chronic Workload Ratio — sports science analogue of our cumulative fatigue model |
| **BioGears** | Open-source whole-body physiology engine; used as the gold standard data source |
| **Digital Twin** | Virtual real-time replica of a physical system, used for monitoring and prediction |
| **Stateful Accumulator** | A variable that remembers its previous value and updates incrementally over time |
| **Phase Space** | Mathematical space where each axis represents a system variable (HR, MAP); a diagnostic visualization |

---

## 📚 Academic & Research References

The physiological models, threshold values, and multi-phasic responses in the digital twin are strictly grounded in peer-reviewed aerospace medicine literature and ISRO mission constraints:

1. **G-Force & Heart Rate Response:** *Scientific Reports (2020)* — Combined effect of HR responses and AGSM effectiveness on G tolerance in a human centrifuge. (Provides the 10 bpm/G steady-state multiplier).
2. **Launch Acceleration Limits:** *Wikipedia / ISRO / LVM3 (2024)* — Maximum acceleration during the ascent phase is limited to 4.0 Gs for crew comfort (Human-rated LVM3).
3. **Microgravity Fluid Shifts & CO:** *Biomedicines (2022)* — The Cardiovascular System in Space. (Details the 35–46% stroke volume rise and stable MAP despite cephalad fluid shift).
4. **Baroreflex Adaptation:** *J. Applied Physiology (2008)* — Dynamic adaptation of cardiac baroreflex sensitivity to prolonged microgravity (16-day STS-107 mission).
5. **Orthostatic Intolerance & Recovery:** *npj Microgravity (2022)* — Computational modeling of orthostatic intolerance for travel to Mars. (Provides the HR ratio of 1.3 vs pre-flight on landing day and dual-exponential recovery times).
6. **Plasma Volume Loss:** *Convertino VA (1996)* & *PMC12419952* — Physiological Changes in the Cardiovascular System During Space Flight. (Validates the 7–20% plasma volume loss in short missions).
7. **Reclined Gx (Chest-to-Back) Re-entry:** *Pollock JL (2021) / Experimental Physiology* — "Oh G: The x, y and z of human physiological responses to acceleration." (Validates the 0.55x CV-equivalent multiplier for 70° reclined crew module seats).
8. **Re-entry Profile:** *Human Physiology / Springer (2015)* — Cosmonauts' tolerance of the chest-back G-loads during ballistic and automatically controlled descents. (Validates the 4.0 G peak during Soyuz-class ballistic re-entry).
9. **Sympathetic Nerve Activity:** *AHA Circulation (2019)* — Impact of prolonged spaceflight on orthostatic tolerance. (Details the ~60% spike in sympathetic tone on landing day).
10. **Exercise in Microgravity:** *Iellamo F (2006) / J Physiol* — Muscle metaboreflex during exercise in microgravity. (Validates that HR/MAP responses to exercise are proportional to 1-G environments).

---

## 📌 Key Numerical Summary (Quick Reference for PPT)

| Parameter | Value | Source |
|---|---|---|
| Tachycardia threshold | > 100 bpm | Clinical standard |
| Hypotension threshold | < 65 mmHg | Critical perfusion threshold |
| CII stable threshold | < 0.15 | Project-defined |
| CII high-risk threshold | > 0.25 | Project-defined |
| Fatigue critical threshold | > 80 / 100 | Project-defined |
| Recovery tolerance | ±10% of baseline | Clinical standard |
| Recovery streak needed | 6 consecutive samples | Project-defined |
| Default Monte Carlo runs | 200 | Configurable (50–500) |
| Synthetic extension target | 48 hours | Mission planning |
| Synthetic time resolution | 1 min intervals | Project-defined |
| Default cardiac output | 5.2 L/min | Average adult male |
| HR noise (Monte Carlo) | σ = 2.0 bpm | Project-defined |
| MAP noise (Monte Carlo) | σ = 1.5 mmHg | Project-defined |
| CO noise (Monte Carlo) | σ = 0.12 L/min | Project-defined |

---

*Built for ISRO Gaganyaan Mission Support — Space Medicine Division*  
*BioGears-inspired cardiovascular monitoring system | Python + Streamlit + Plotly*
