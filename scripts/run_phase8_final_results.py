#!/usr/bin/env python3
"""
scripts/run_phase8_final_results.py — Phase 8: final paper tables and figures.


"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "phase8_final"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_LABELS = {
    "mlp": "MLP",
    "gcn": "GCN",
    "gat": "GAT",
    "stgcn": "STGCN",
    "stgat_nogate": "STGAT-NoGate",
    "resiligraph_stgat": "ResiliGraph-STGAT",
}

MODEL_ORDER = [
    "mlp",
    "gcn",
    "gat",
    "stgcn",
    "stgat_nogate",
    "resiligraph_stgat",
]


def safe_float(v, default=np.nan):
    try:
        if v is None or pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def metric(row, name, default=np.nan):
    return safe_float(row[name], default) if name in row else default


def build_table1():
    rows = []

    for key in MODEL_ORDER:
        p = ROOT / "results" / "phase3" / key / "test_metrics.csv"

        if not p.exists():
            continue

        r = pd.read_csv(p).iloc[0]

        rows.append(
            {
                "Model": MODEL_LABELS[key],
                "model_key": key,
                "Fault AUC": metric(r, "fault_risk_auc"),
                "Congestion AUC": metric(r, "congestion_risk_auc"),
                "Voltage AUC": metric(r, "voltage_violation_auc"),
                "Mean AUC": metric(r, "macro_auc"),
                "Mean F1": metric(r, "macro_f1"),
                "Mean AUPRC": metric(r, "macro_auprc"),
                "Mean Recall": metric(r, "macro_recall"),
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        df.to_csv(OUT / "table1_model_comparison.csv", index=False)
        return df

    if df["Mean AUC"].isna().any():
        df["Mean AUC"] = df[["Fault AUC", "Congestion AUC", "Voltage AUC"]].mean(
            axis=1
        )

    multiseed_csv = ROOT / "results" / "phase3" / "multiseed_all_models.csv"

    if multiseed_csv.exists():
        ms = pd.read_csv(multiseed_csv)

        for i, row in df.iterrows():
            key = row["model_key"]
            sub = ms[ms["model"] == key]

            if sub.empty:
                continue

            s = sub.iloc[0]

            df.loc[i, "Mean F1 (mean)"] = safe_float(s.get("macro_f1_mean"))
            df.loc[i, "Mean F1 (std)"] = safe_float(s.get("macro_f1_std"))
            df.loc[i, "Mean AUC (mean)"] = safe_float(s.get("macro_auc_mean"))
            df.loc[i, "Mean AUC (std)"] = safe_float(s.get("macro_auc_std"))
            df.loc[i, "Mean AUPRC (mean)"] = safe_float(s.get("macro_auprc_mean"))
            df.loc[i, "Mean AUPRC (std)"] = safe_float(s.get("macro_auprc_std"))
            df.loc[i, "Mean Recall (mean)"] = safe_float(s.get("macro_recall_mean"))
            df.loc[i, "Mean Recall (std)"] = safe_float(s.get("macro_recall_std"))

    for col in [
        "Mean F1 (mean)",
        "Mean AUC (mean)",
        "Mean AUPRC (mean)",
        "Mean Recall (mean)",
    ]:
        if col in df.columns:
            base_col = col.replace(" (mean)", "")
            df[col] = df[col].fillna(df[base_col])

    df = df.drop(columns=["model_key"])
    df.to_csv(OUT / "table1_model_comparison.csv", index=False)
    return df


def build_table2():
    csv_path = ROOT / "results" / "phase5_renewable" / "renewable_comparison.csv"

    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(
            [
                {
                    "Configuration": "Phase 5 not generated yet",
                    "Fault AUC": np.nan,
                    "Congestion AUC": np.nan,
                    "Voltage AUC": np.nan,
                    "Mean AUC": np.nan,
                    "Mean F1": np.nan,
                    "Mean AUPRC": np.nan,
                }
            ]
        )

    df.to_csv(OUT / "table2_renewable_comparison.csv", index=False)
    return df


def build_table3():
    p = ROOT / "results" / "phase6" / "copilot_recommendations.csv"

    if p.exists():
        df = pd.read_csv(p)

        keep = [
            c
            for c in [
                "timestep",
                "bus",
                "risk_type",
                "fault_prob",
                "congestion_prob",
                "voltage_prob",
                "recommended_action",
                "explanation",
            ]
            if c in df.columns
        ]

        examples = []

        for risk in ["fault", "congestion", "voltage", "normal"]:
            if "risk_type" not in df.columns:
                break

            sub = df[df["risk_type"] == risk]

            if len(sub) > 0:
                if risk == "normal":
                    row = sub.sort_values("fault_prob").iloc[0]
                else:
                    col_map = {
                        "fault": "fault_prob",
                        "congestion": "congestion_prob",
                        "voltage": "voltage_prob",
                    }
                    row = sub.sort_values(col_map[risk], ascending=False).iloc[0]

                examples.append(row[keep].to_dict())

        outdf = pd.DataFrame(examples) if examples else df.head(8)[keep]

    else:
        outdf = pd.DataFrame(
            [{"note": "Phase 6 copilot_recommendations.csv not generated yet"}]
        )

    outdf.to_csv(OUT / "table3_copilot_examples.csv", index=False)
    return outdf


def build_table4():
    p = ROOT / "results" / "phase7_rl" / "rl_summary.json"

    if p.exists():
        s = json.loads(p.read_text())

        rule_r = safe_float(s.get("rule_final_avg_reward"))
        dqn_r = safe_float(s.get("dqn_final_avg_reward"))
        impr = safe_float(s.get("improvement"))

        pct = round(100.0 * impr / abs(rule_r), 2) if rule_r and rule_r != 0 else np.nan

        df = pd.DataFrame(
            [
                {
                    "Policy": "Rule-based",
                    "Mean Reward": rule_r,
                    "Improvement %": "—",
                },
                {
                    "Policy": "DQN",
                    "Mean Reward": dqn_r,
                    "Improvement %": f"+{pct:.1f}%" if pd.notna(pct) else "",
                },
                {
                    "Policy": "Δ DQN-Rule",
                    "Mean Reward": impr,
                    "Improvement %": "",
                },
            ]
        )

    else:
        df = pd.DataFrame(
            [{"Policy": "Phase 7 not generated yet", "Mean Reward": np.nan}]
        )

    df.to_csv(OUT / "table4_rl_performance.csv", index=False)
    return df


def build_table5():
    path = ROOT / "results" / "phase4" / "top_critical_lines.csv"

    if path.exists():
        df = pd.read_csv(path).head(10)
    else:
        df = pd.DataFrame(
            [
                {
                    "rank": "Phase 4 not generated yet",
                    "from_bus_name": None,
                    "to_bus_name": None,
                    "attn_mean": None,
                    "attn_std": None,
                    "loading_pct_mean": None,
                    "loading_pct_std": None,
                }
            ]
        )

    df.to_csv(OUT / "table5_explainability_critical_lines.csv", index=False)
    return df


def build_table6():
    path = ROOT / "results" / "phase7_rl" / "rl_summary.json"

    if not path.exists():
        df = pd.DataFrame([{"action": "missing", "count": 0}])
        df.to_csv(OUT / "table6_rl_action_distribution.csv", index=False)
        return df

    data = json.loads(path.read_text())
    dist = data.get("action_distribution", {})

    df = pd.DataFrame([{"action": k, "count": v} for k, v in dist.items()])

    if not df.empty:
        df = df.sort_values("count", ascending=False)

    df.to_csv(OUT / "table6_rl_action_distribution.csv", index=False)
    return df


def build_table7():
    path = ROOT / "results" / "phase11" / "stress_benchmark_summary.csv"

    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(
            [{"note": "Phase 11 stress benchmark not generated yet"}]
        )

    df.to_csv(OUT / "table7_stress_regime_performance.csv", index=False)
    return df


def build_table8_best_by_stress():
    path = ROOT / "results" / "phase11" / "stress_best_by_level.csv"

    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(
            [{"note": "Phase 11 best-by-stress table not generated yet"}]
        )

    df.to_csv(OUT / "table8_best_by_stress.csv", index=False)
    return df


def figure1_architecture():
    labels = [
        "OpenDSS / IEEE 33",
        "Temporal Graphs",
        "STGCN / STGAT",
        "Explainability",
        "Renewable Features",
        "Copilot Actions",
        "DQN RL",
    ]

    fig, ax = plt.subplots(figsize=(13, 3.2))
    ax.axis("off")

    for i, label in enumerate(labels):
        x = i * 1.8

        ax.add_patch(
            plt.Rectangle((x, 0.8), 1.45, 0.75, fill=False, linewidth=1.8)
        )

        ax.text(
            x + 0.725,
            1.175,
            label,
            ha="center",
            va="center",
            fontsize=9,
        )

        if i < len(labels) - 1:
            ax.annotate(
                "",
                xy=(x + 1.75, 1.175),
                xytext=(x + 1.45, 1.175),
                arrowprops=dict(arrowstyle="->", lw=1.5),
            )

    ax.set_xlim(-0.1, len(labels) * 1.8 - 0.2)
    ax.set_ylim(0.4, 1.9)

    ax.set_title(
        "AI Grid Resilience Copilot — System Architecture",
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(OUT / "figure1_architecture.png", dpi=150, bbox_inches="tight")
    plt.close()


def figure2_graph():
    graph_path = ROOT / "data" / "processed" / "phase1" / "graph.pt"

    fig, ax = plt.subplots(figsize=(8, 5))

    if graph_path.exists():
        g = torch.load(graph_path, map_location="cpu", weights_only=False)
        edge_index = g.edge_index.numpy()

        G = nx.Graph()
        G.add_edges_from(edge_index.T.tolist())

        pos = nx.spring_layout(G, seed=42)

        nx.draw(
            G,
            pos,
            with_labels=True,
            node_size=350,
            font_size=7,
            ax=ax,
        )

    else:
        ax.text(0.5, 0.5, "Phase 1 graph.pt not found", ha="center")

    ax.set_title("IEEE 33-bus Feeder Graph")

    plt.tight_layout()
    plt.savefig(OUT / "figure2_ieee33_graph.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_values(df: pd.DataFrame, preferred_col: str, fallback_col: str):
    if preferred_col in df.columns:
        vals = pd.to_numeric(df[preferred_col], errors="coerce")
        if vals.notna().any():
            return vals

    if fallback_col in df.columns:
        return pd.to_numeric(df[fallback_col], errors="coerce")

    return pd.Series([np.nan] * len(df))


def figure3_performance(df):
    fig, ax = plt.subplots(figsize=(9, 4.5))

    if not df.empty and "Model" in df.columns:
        vals = _plot_values(df, "Mean F1 (mean)", "Mean F1").fillna(0.0)

        ax.bar(df["Model"], vals)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Macro F1")
        ax.tick_params(axis="x", rotation=25)

    else:
        ax.text(0.5, 0.5, "No Phase 3 metrics found", ha="center")

    ax.set_title("Temporal Risk Prediction Performance")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "figure3_model_performance.png", dpi=150, bbox_inches="tight")
    plt.close()


def figure4_stress_performance(table7):
    fig, ax = plt.subplots(figsize=(9, 5))

    required = {"stress_scale", "model", "macro_auprc_mean"}

    if not table7.empty and required.issubset(table7.columns):
        plot_df = table7.copy()
        plot_df["macro_auprc_mean"] = pd.to_numeric(
            plot_df["macro_auprc_mean"],
            errors="coerce",
        )

        for model in plot_df["model"].dropna().unique():
            sub = plot_df[plot_df["model"] == model].sort_values("stress_scale")
            label = MODEL_LABELS.get(model, model)

            ax.plot(
                sub["stress_scale"],
                sub["macro_auprc_mean"],
                marker="o",
                label=label,
            )

        ax.set_xlabel("Stress scale")
        ax.set_ylabel("Macro AUPRC")
        ax.set_ylim(0.75, 1.02)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    else:
        ax.text(0.5, 0.5, "Phase 11 stress benchmark not found", ha="center")

    ax.set_title("Stress-Regime AUPRC Performance")

    plt.tight_layout()
    plt.savefig(OUT / "figure4_stress_regime_auprc.png", dpi=150, bbox_inches="tight")
    plt.close()


def figure_summary(table1, table2, table4):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    if not table1.empty and "Model" in table1.columns:
        vals = _plot_values(table1, "Mean F1 (mean)", "Mean F1").fillna(0.0)

        axes[0].bar(table1["Model"], vals)
        axes[0].tick_params(axis="x", rotation=25)
        axes[0].set_ylim(0, 1.05)

    axes[0].set_title("Model Macro F1")
    axes[0].grid(axis="y", alpha=0.3)

    if (
        "Configuration" in table2.columns
        and "Mean F1" in table2.columns
        and table2["Mean F1"].notna().any()
    ):
        short_labels = [
            "w/o Renewable" if "Without" in str(c) else "w/ Renewable"
            for c in table2["Configuration"]
        ]

        vals = pd.to_numeric(table2["Mean F1"], errors="coerce").fillna(0.0)

        axes[1].bar(short_labels, vals)
        axes[1].tick_params(axis="x", rotation=10)
        axes[1].set_ylim(0, 1.05)

    axes[1].set_title("Renewable-aware Comparison")
    axes[1].grid(axis="y", alpha=0.3)

    if "Policy" in table4.columns and "Mean Reward" in table4.columns:
        plot_t4 = table4[table4["Policy"].isin(["Rule-based", "DQN"])].copy()

        if not plot_t4.empty:
            plot_t4["Mean Reward"] = pd.to_numeric(
                plot_t4["Mean Reward"],
                errors="coerce",
            )

            axes[2].bar(plot_t4["Policy"], plot_t4["Mean Reward"].fillna(0.0))
            axes[2].tick_params(axis="x", rotation=10)

    axes[2].set_title("RL Policy Reward")
    axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "paper_final_summary.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    print("=" * 70)
    print("Phase 8 — Final Paper Results")
    print("=" * 70)

    table1 = build_table1()
    print("\nTable 1\n", table1.to_string(index=False))

    table2 = build_table2()
    print("\nTable 2\n", table2.to_string(index=False))

    table3 = build_table3()
    print("\nTable 3\n", table3.head().to_string(index=False))

    table4 = build_table4()
    print("\nTable 4\n", table4.to_string(index=False))

    table5 = build_table5()
    print("\nTable 5\n", table5.to_string(index=False))

    table6 = build_table6()
    print("\nTable 6\n", table6.to_string(index=False))

    table7 = build_table7()
    print("\nTable 7 — Stress-Regime Performance\n", table7.head(12).to_string(index=False))

    table8 = build_table8_best_by_stress()
    print("\nTable 8 — Best Model by Stress Level\n", table8.to_string(index=False))

    figure1_architecture()
    figure2_graph()
    figure3_performance(table1)
    figure4_stress_performance(table7)
    figure_summary(table1, table2, table4)

    print("\nSaved Phase 8 outputs to", OUT)


if __name__ == "__main__":
    main()