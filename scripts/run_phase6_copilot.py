import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.copilot.rule_based_actions import recommend_action


def main():
    print("=" * 70)
    print("Phase 6 — Rule-Based Copilot Recommendations")
    print("=" * 70)

    out_dir = PROJECT_ROOT / "results" / "phase6"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = (
        PROJECT_ROOT
        / "results"
        / "phase3"
        / "resiligraph_stgat"
        / "test_predictions.csv"
    )

    if not pred_path.exists():
        print(f"Missing predictions file: {pred_path}")
        print("You need test_predictions.csv with fault/congestion/voltage probabilities.")
        return

    df = pd.read_csv(pred_path)

    required = ["fault_prob", "congestion_prob", "voltage_prob"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        print("Missing required columns:", missing)
        print("Your prediction file must contain:")
        print(required)
        return

    rows = []

    for i, row in df.iterrows():
        fault_prob = float(row["fault_prob"])
        congestion_prob = float(row["congestion_prob"])
        voltage_prob = float(row["voltage_prob"])

        rec = recommend_action(
            fault_prob=fault_prob,
            congestion_prob=congestion_prob,
            voltage_prob=voltage_prob,
            threshold=0.5,
        )

        rows.append({
            "timestep": row.get("timestep", i),
            "bus": row.get("bus", "system"),
            "fault_prob": fault_prob,
            "congestion_prob": congestion_prob,
            "voltage_prob": voltage_prob,
            "risk_type": rec["risk_type"],
            "recommended_action": rec["recommended_action"],
            "explanation": rec["explanation"],
        })

    out = pd.DataFrame(rows)

    out["max_risk"] = out[
        ["fault_prob", "congestion_prob", "voltage_prob"]
    ].max(axis=1)

    # ------------------------------------------------------------
    # Build diverse paper-ready examples
    # ------------------------------------------------------------
    paper_rows = []

    for risk in ["fault", "congestion", "voltage", "normal"]:
        subset = out[out["risk_type"] == risk]

        if len(subset) > 0:
            if risk == "normal":
                chosen = subset.sort_values("max_risk", ascending=True).head(1)
            else:
                chosen = subset.sort_values("max_risk", ascending=False).head(1)

            paper_rows.append(chosen)

    if paper_rows:
        paper_table = pd.concat(paper_rows, ignore_index=True)
    else:
        paper_table = out.sort_values("max_risk", ascending=False).head(5)

    full_path = out_dir / "copilot_recommendations_full.csv"
    out.to_csv(full_path, index=False)

    save_path = out_dir / "copilot_recommendations.csv"
    paper_table.to_csv(save_path, index=False)

    print(f"Saved full recommendations: {full_path}")
    print(f"Saved paper table: {save_path}")
    print(paper_table.to_string(index=False))


if __name__ == "__main__":
    main()