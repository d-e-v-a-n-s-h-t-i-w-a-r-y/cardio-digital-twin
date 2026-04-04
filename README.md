# Cardiovascular Adaptation Digital Twin using BioGears

A beginner-friendly Python project that simulates astronaut cardiovascular adaptation during a space mission, analyzes cardiovascular risk, and visualizes results in a Streamlit dashboard.

## What it does

- Loads simulated BioGears-style cardiovascular time-series data
- Simulates mission events:
  - microgravity fluid shift
  - exercise workload
  - re-entry stress
  - recovery phase
- Applies probabilistic modeling:
  - Normal distribution for HR and MAP variability
  - Poisson distribution for random stress events
  - Log-normal distribution for exercise intensity
- Detects cardiovascular risks:
  - hypotension
  - tachycardia
  - instability index
  - recovery time
  - workload index
- Runs Monte Carlo simulations
- Shows an interactive dashboard in Streamlit

## Folder structure

```text
cardio_digital_twin/
├── biogears_data/
│   └── cardiovascular_data.csv
├── simulation/
├── analytics/
├── visualization/
└── main.py
```

## Setup on macOS

### 1) Create and activate a virtual environment

```bash
cd cardio_digital_twin
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Run the dashboard

```bash
streamlit run visualization/dashboard.py
```

### 4) Optional: run the command-line demo

```bash
python main.py
```

## Notes

- The included CSV is synthetic sample data shaped like BioGears output.
- You can replace it with real BioGears exports if needed, as long as the column names remain compatible.
