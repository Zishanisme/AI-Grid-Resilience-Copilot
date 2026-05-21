#!/usr/bin/env python3
"""
scripts/run_multiseed_phase3.py

Reviewer-safe multi-seed evaluation for all Phase 3 models.

Runs:
- mlp
- gcn
- gat
- stgcn
- stgat_nogate
- resiligraph_stgat

Across seeds:
- 11
- 22
- 33

Outputs:
results/phase3/multiseed_all_models.json
results/phase3/multiseed_all_models.csv
results/phase3/multiseed_all_model_runs.csv
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEEDS = [11, 22, 33]

MODELS = [
    "mlp",
    "gcn",
    "gat",
    "stgcn",
    "stgat_nogate",
    "resiligraph_stgat",
]

EPOCHS = 400
SEQ_LEN = 20
LR = "8e-4"
PATIENCE = 60

SUMMARY_METRICS = [
    "macro_f1",
    "macro_auc",
    "macro_auprc",
    "macro_recall",
]

_venv_win = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
_venv_unix = PROJECT_ROOT / ".venv" / "bin" / "python"

PYTHON_EXE = (
    str(_venv_win)
    if _venv_win.exists()
    else str(_venv_unix)
    if _venv_unix.exists()
    else sys.executable
)


def run_one_model_seed(model: str, seed: int) -> dict | None:
    print("\n" + "=" * 80)
    print(f"RUNNING MODEL={model}  SEED={seed}")
    print("=" * 80)

    cmd = [
        PYTHON_EXE,
        "scripts/run_phase3_train.py",
        "--model",
        model,
        "--epochs",
        str(EPOCHS),
        "--seq_len",
        str(SEQ_LEN),
        "--lr",
        LR,
        "--patience",
        str(PATIENCE),
        "--seed",
        str(seed),
    ]

    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)

    metrics_path = (
        PROJECT_ROOT
        / "results"
        / "phase3"
        / model
        / "test_metrics.csv"
    )

    if not metrics_path.exists():
        print(f"[WARN] Missing metrics file: {metrics_path}")
        return None

    df = pd.read_csv(metrics_path)

    if df.empty:
        print(f"[WARN] Empty metrics file: {metrics_path}")
        return None

    row = df.iloc[0].to_dict()
    row["model"] = model
    row["seed"] = seed

    print(
        f"{model} seed {seed} — "
        f"macro_f1={float(row.get('macro_f1', 0.0)):.4f}  "
        f"macro_auc={float(row.get('macro_auc', 0.0)):.4f}  "
        f"macro_auprc={float(row.get('macro_auprc', 0.0)):.4f}  "
        f"macro_recall={float(row.get('macro_recall', 0.0)):.4f}"
    )

    return row


def summarize_model(model: str, rows: list[dict]) -> dict:
    model_rows = [r for r in rows if r["model"] == model]

    summary = {
        "model": model,
        "n_runs": len(model_rows),
    }

    for metric in SUMMARY_METRICS:
        vals = [
            float(r[metric])
            for r in model_rows
            if metric in r and pd.notna(r[metric])
        ]

        if len(vals) >= 2:
            summary[f"{metric}_mean"] = round(statistics.mean(vals), 4)
            summary[f"{metric}_std"] = round(statistics.stdev(vals), 4)
            summary[f"{metric}_min"] = round(min(vals), 4)
            summary[f"{metric}_max"] = round(max(vals), 4)
        elif len(vals) == 1:
            summary[f"{metric}_mean"] = round(vals[0], 4)
            summary[f"{metric}_std"] = 0.0
            summary[f"{metric}_min"] = round(vals[0], 4)
            summary[f"{metric}_max"] = round(vals[0], 4)
        else:
            summary[f"{metric}_mean"] = None
            summary[f"{metric}_std"] = None
            summary[f"{metric}_min"] = None
            summary[f"{metric}_max"] = None

    return summary


def main() -> None:
    all_rows: list[dict] = []

    for model in MODELS:
        for seed in SEEDS:
            row = run_one_model_seed(model=model, seed=seed)
            if row is not None:
                all_rows.append(row)

    summaries = [
        summarize_model(model=model, rows=all_rows)
        for model in MODELS
    ]

    out_root = PROJECT_ROOT / "results" / "phase3"
    out_root.mkdir(parents=True, exist_ok=True)

    json_path = out_root / "multiseed_all_models.json"
    csv_path = out_root / "multiseed_all_models.csv"
    runs_csv_path = out_root / "multiseed_all_model_runs.csv"

    output = {
        "seeds": SEEDS,
        "models": MODELS,
        "runs": all_rows,
        "summary": summaries,
    }

    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    pd.DataFrame(summaries).to_csv(csv_path, index=False)
    pd.DataFrame(all_rows).to_csv(runs_csv_path, index=False)

    print("\n" + "=" * 80)
    print("MULTI-SEED SUMMARY — ALL MODELS")
    print("=" * 80)

    summary_df = pd.DataFrame(summaries)

    display_cols = [
        "model",
        "n_runs",
        "macro_f1_mean",
        "macro_f1_std",
        "macro_auc_mean",
        "macro_auc_std",
        "macro_auprc_mean",
        "macro_auprc_std",
        "macro_recall_mean",
        "macro_recall_std",
    ]

    print(summary_df[display_cols].to_string(index=False))

    print(f"\nSaved summary JSON → {json_path}")
    print(f"Saved summary CSV  → {csv_path}")
    print(f"Saved run CSV      → {runs_csv_path}")


if __name__ == "__main__":
    main()