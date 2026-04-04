from __future__ import annotations

from pathlib import Path

from simulation.digital_twin_model import CardiovascularDigitalTwin
from simulation.event_simulator import simulate_mission
from simulation.load_biogears_data import load_biogears_data


def main() -> None:
    root = Path(__file__).resolve().parent
    csv_path = root / "biogears_data" / "cardiovascular_data.csv"
    base_df = load_biogears_data(csv_path)
    simulated_df = simulate_mission(base_df, seed=42)
    twin = CardiovascularDigitalTwin(base_df)
    result = twin.analyze(simulated_df)

    print("Cardiovascular Adaptation Digital Twin")
    print("-------------------------------------")
    print(f"Rows simulated: {len(result.data)}")
    print(f"Instability index: {result.instability_index:.3f}")
    print(f"Hypotension count: {result.summary['hypotension_count']}")
    print(f"Tachycardia count: {result.summary['tachycardia_count']}")
    print(f"Recovery time (min): {result.recovery_time_min}")
    print(f"Mean workload index: {result.summary['mean_workload_index']:.2f}")


if __name__ == "__main__":
    main()
