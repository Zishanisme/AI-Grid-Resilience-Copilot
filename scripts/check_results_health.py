#!/usr/bin/env python3
from pathlib import Path
import json
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
TASKS = ["fault_risk", "congestion_risk", "voltage_violation"]

print("="*70)
print("AI Grid Resilience — Results Health Check")
print("="*70)

# Label balance
for split in ["train", "val", "test"]:
    p = ROOT / f"data/processed/phase2/{split}_graphs.pt"
    if not p.exists():
        print(f"MISSING: {p}")
        continue
    graphs = torch.load(p, map_location="cpu", weights_only=False)
    y = torch.cat([g.y for g in graphs], dim=0)
    print(f"\n{split.upper()} label positive rate:")
    for i, t in enumerate(TASKS):
        print(f"  {t:<20} {y[:, i].float().mean().item():.4f}  positives={int(y[:, i].sum().item())}/{y.shape[0]}")

# Phase 3 metrics
print("\nPhase 3 test metrics:")
rows = []
for model_dir in sorted((ROOT / "results/phase3").glob("*/test_metrics.csv")):
    model = model_dir.parent.name
    df = pd.read_csv(model_dir)
    row = df.iloc[0].to_dict()
    rows.append({
        "model": model,
        "macro_f1": row.get("macro_f1"),
        "macro_auc": row.get("macro_auc"),
        "macro_auprc": row.get("macro_auprc"),
        "fault_auc": row.get("fault_risk_auc"),
        "congestion_auc": row.get("congestion_risk_auc"),
        "voltage_auc": row.get("voltage_violation_auc"),
    })
if rows:
    print(pd.DataFrame(rows).sort_values("macro_f1", ascending=False).to_string(index=False))
else:
    print("  No results/phase3/*/test_metrics.csv files found.")
