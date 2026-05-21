#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IN = PROJECT_ROOT / "results" / "phase11" / "stress_benchmark.csv"
OUT = PROJECT_ROOT / "results" / "phase11"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(IN)

metrics = [
    "macro_f1",
    "macro_auc",
    "macro_auprc",
    "macro_recall",
]

pretty = {
    "macro_f1": "Macro F1",
    "macro_auc": "Macro AUC",
    "macro_auprc": "Macro AUPRC",
    "macro_recall": "Macro Recall",
}

for metric in metrics:
    plt.figure(figsize=(8, 5))

    for model in df["model"].unique():
        sub = df[df["model"] == model]
        plt.plot(
            sub["stress_scale"],
            sub[metric],
            marker="o",
            label=model,
        )

    plt.xlabel("Renewable/Cascading Stress Scale")
    plt.ylabel(pretty[metric])
    plt.title(f"Stress Robustness — {pretty[metric]}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = OUT / f"stress_{metric}.png"
    plt.savefig(path, dpi=300)
    plt.close()

summary = (
    df.groupby("model")[metrics]
    .mean()
    .reset_index()
    .sort_values("macro_f1", ascending=False)
)

summary.to_csv(OUT / "stress_summary_mean.csv", index=False)

best_by_stress = []

for stress in sorted(df["stress_scale"].unique()):
    sub = df[df["stress_scale"] == stress]

    best_f1 = sub.sort_values("macro_f1", ascending=False).iloc[0]
    best_auc = sub.sort_values("macro_auc", ascending=False).iloc[0]
    best_auprc = sub.sort_values("macro_auprc", ascending=False).iloc[0]

    best_by_stress.append({
        "stress_scale": stress,
        "best_f1_model": best_f1["model"],
        "best_f1": best_f1["macro_f1"],
        "best_auc_model": best_auc["model"],
        "best_auc": best_auc["macro_auc"],
        "best_auprc_model": best_auprc["model"],
        "best_auprc": best_auprc["macro_auprc"],
    })

best_df = pd.DataFrame(best_by_stress)
best_df.to_csv(OUT / "stress_best_by_level.csv", index=False)

print("Saved Phase 11 plots and tables to:", OUT)
print("\nMean stress summary:")
print(summary.to_string(index=False))

print("\nBest model by stress level:")
print(best_df.to_string(index=False))