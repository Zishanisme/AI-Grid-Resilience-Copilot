#!/usr/bin/env python3
"""
scripts/run_phase4_explainability.py

Phase 4: Attention Explainability + Physical Validation.

Adds:
- GAT layer-1 and layer-2 attention aggregation
- top critical lines
- top risk buses
- attention feeder visualization
- Spearman correlation between attention score and loading_pct
- reverse-normalization of loading_pct for interpretable CSV tables

Important interpretation:
Because ResiliGraph-STGAT uses edge_attr with loading_pct inside GATConv(edge_dim=4),
the attention-loading correlation validates that the edge-conditioned attention
mechanism is using physical loading information. It should not be framed as an
independent discovery.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.st_gnn import STGAT_GRU
from src.train.utils import load_checkpoint, make_sequences


SEQ_LEN = 20


def ieee33_positions() -> dict:
    pos = {}

    for i in range(18):
        pos[i] = (i * 1.0, 0.0)

    for j, b in enumerate([18, 19, 20, 21]):
        pos[b] = (1.0 + j, -1.5)

    for j, b in enumerate([22, 23, 24]):
        pos[b] = (5.0 + j, -1.5)

    for j, b in enumerate([25, 26, 27]):
        pos[b] = (8.0 + j, 1.5)

    for j, b in enumerate([28, 29, 30, 31, 32]):
        pos[b] = (11.0 + j, 1.5)

    return pos


def _edge_loading_map(
    g,
    edge_mean_loading: float,
    edge_std_loading: float,
) -> dict[tuple[int, int], float]:
    """
    Build mapping:
    (src, dst) -> raw loading_pct

    Assumes build_graph stores edge_attr columns:
    [r_ohm, x_ohm, length, loading_pct]

    Note:
    test graphs are normalized after apply_feature_stats, so edge_attr[:, 3]
    is z-scored. This function reverse-normalizes loading_pct before saving.
    """

    loading_map = {}

    if not hasattr(g, "edge_attr") or g.edge_attr is None:
        return loading_map

    edge_index = g.edge_index.detach().cpu().numpy()
    edge_attr = g.edge_attr.detach().cpu().numpy()

    if edge_attr.ndim != 2 or edge_attr.shape[1] < 4:
        return loading_map

    n_edges = min(edge_index.shape[1], edge_attr.shape[0])

    for i in range(n_edges):
        src = int(edge_index[0, i])
        dst = int(edge_index[1, i])

        z_loading = float(edge_attr[i, 3])
        loading = z_loading * edge_std_loading + edge_mean_loading

        loading_map[(src, dst)] = float(loading)

    return loading_map


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float | None]:
    """
    Spearman correlation with scipy if available.
    Fallback: pandas rank correlation without p-value.
    """

    if len(x) < 3:
        return float("nan"), None

    try:
        from scipy.stats import spearmanr

        rho, p_value = spearmanr(x, y)
        return float(rho), float(p_value)

    except Exception:
        df = pd.DataFrame({"x": x, "y": y})
        rho = df["x"].corr(df["y"], method="spearman")
        return float(rho), None


def _build_edge_df(
    store: dict,
    loading_store: dict,
    bnames,
) -> pd.DataFrame:
    rows = []

    for (src, dst), scores in store.items():
        if len(scores) == 0:
            continue

        sn = bnames[src] if bnames and src < len(bnames) else str(src)
        dn = bnames[dst] if bnames and dst < len(bnames) else str(dst)

        loadings = loading_store.get((src, dst), [])

        rows.append(
            {
                "from_bus": int(src),
                "to_bus": int(dst),
                "from_bus_name": sn,
                "to_bus_name": dn,
                "attn_mean": float(np.mean(scores)),
                "attn_std": float(np.std(scores)),
                "loading_pct_mean": (
                    float(np.mean(loadings)) if len(loadings) > 0 else np.nan
                ),
                "loading_pct_std": (
                    float(np.std(loadings)) if len(loadings) > 0 else np.nan
                ),
                "n_sequences": len(scores),
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df = df.sort_values("attn_mean", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))

    return df


def _load_edge_loading_stats(data_dir: Path) -> tuple[float, float]:
    """
    Load edge feature normalization stats and return mean/std
    for edge_attr[:, 3] = loading_pct.
    """

    stats_path = data_dir / "feature_stats.pt"

    if not stats_path.exists():
        print(
            "[WARN] feature_stats.pt not found. "
            "loading_pct_mean will remain normalized."
        )
        return 0.0, 1.0

    stats = torch.load(stats_path, weights_only=False)

    if "edge_mean" not in stats or "edge_std" not in stats:
        print(
            "[WARN] edge_mean/edge_std not found in feature_stats.pt. "
            "loading_pct_mean will remain normalized."
        )
        return 0.0, 1.0

    edge_mean = stats["edge_mean"]
    edge_std = stats["edge_std"]

    if len(edge_mean) < 4 or len(edge_std) < 4:
        print(
            "[WARN] edge normalization stats have fewer than 4 columns. "
            "loading_pct_mean will remain normalized."
        )
        return 0.0, 1.0

    edge_mean_loading = float(edge_mean[3].item())
    edge_std_loading = float(edge_std[3].item())

    if abs(edge_std_loading) < 1e-12:
        edge_std_loading = 1.0

    return edge_mean_loading, edge_std_loading


def main():
    print("=" * 70)
    print("Phase 4 — Attention Explainability + Loading Correlation")
    print("=" * 70)

    data_dir = PROJECT_ROOT / "data/processed/phase2"
    ckpt_path = PROJECT_ROOT / "results/phase3/resiligraph_stgat/best_model.pt"
    meta_path = PROJECT_ROOT / "results/phase3/resiligraph_stgat/model_meta.json"
    out_dir = PROJECT_ROOT / "results/phase4"

    out_dir.mkdir(parents=True, exist_ok=True)

    edge_mean_loading, edge_std_loading = _load_edge_loading_stats(data_dir)

    print(
        "Using loading_pct reverse-normalization: "
        f"mean={edge_mean_loading:.6f}, std={edge_std_loading:.6f}"
    )

    test_graphs = torch.load(data_dir / "test_graphs.pt", weights_only=False)
    test_seq = make_sequences(test_graphs, SEQ_LEN)

    if not test_seq:
        raise RuntimeError("No test sequences available. Check data and SEQ_LEN.")

    bus_names_list = getattr(test_seq[0][-1], "bus_names", None)

    in_dim = test_seq[0][-1].x.shape[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if meta_path.exists():
        meta = json.loads(meta_path.read_text())

        model = STGAT_GRU(
            in_dim=meta.get("in_dim", in_dim),
            hidden_dim=meta.get("hidden_dim", 64),
            heads=meta.get("heads", 2),
        )

    else:
        model = STGAT_GRU(in_dim=in_dim)

    model = load_checkpoint(model, ckpt_path, device)
    model.eval()

    edge_attn_g1 = defaultdict(list)
    edge_attn_g2 = defaultdict(list)

    edge_loading_g1 = defaultdict(list)
    edge_loading_g2 = defaultdict(list)

    all_probs = []

    print(f"Aggregating attention over {len(test_seq)} test sequences...")

    for seq in test_seq:
        seq_dev = [g.to(device) for g in seq]

        # Attention weights stored by the model come from the final graph
        # encoded in the sequence, so loading_pct is read from seq[-1].
        g_last_cpu = seq[-1].cpu()

        loading_map = _edge_loading_map(
            g_last_cpu,
            edge_mean_loading=edge_mean_loading,
            edge_std_loading=edge_std_loading,
        )

        with torch.no_grad():
            logits = model(seq_dev)
            probs = torch.sigmoid(logits).detach().cpu().numpy()

        all_probs.append(probs)

        if model.last_attn_weights is None:
            continue

        for layer_key, attn_store, loading_store in [
            ("gat1", edge_attn_g1, edge_loading_g1),
            ("gat2", edge_attn_g2, edge_loading_g2),
        ]:
            if layer_key not in model.last_attn_weights:
                continue

            attn_ei, attn_w = model.last_attn_weights[layer_key]

            attn_ei = attn_ei.detach().cpu().numpy()
            attn_w = attn_w.detach().cpu().numpy()

            if attn_w.ndim == 2:
                attn_w = attn_w.mean(axis=1)

            for i, (src, dst) in enumerate(attn_ei.T):
                src = int(src)
                dst = int(dst)

                # PyG GATConv may return attention on added self-loops.
                # These are not physical feeder lines, so exclude them.
                if src == dst:
                    continue

                edge = (src, dst)

                attn_store[edge].append(float(attn_w[i]))

                if edge in loading_map:
                    loading_store[edge].append(float(loading_map[edge]))

    if not edge_attn_g1:
        raise RuntimeError("No attention weights captured. Check model.last_attn_weights.")

    edge_df_g1 = _build_edge_df(edge_attn_g1, edge_loading_g1, bus_names_list)
    edge_df_g2 = _build_edge_df(edge_attn_g2, edge_loading_g2, bus_names_list)

    full_cols = [
        "rank",
        "from_bus",
        "to_bus",
        "from_bus_name",
        "to_bus_name",
        "attn_mean",
        "attn_std",
        "loading_pct_mean",
        "loading_pct_std",
        "n_sequences",
    ]

    top_cols = [
        "rank",
        "from_bus_name",
        "to_bus_name",
        "attn_mean",
        "attn_std",
        "loading_pct_mean",
        "loading_pct_std",
        "n_sequences",
    ]

    edge_df_g1[full_cols].to_csv(
        out_dir / "edge_attention_loading_layer1.csv",
        index=False,
    )

    edge_df_g2[full_cols].to_csv(
        out_dir / "edge_attention_loading_layer2.csv",
        index=False,
    )

    edge_df_g1[top_cols].head(20).to_csv(
        out_dir / "top_critical_lines.csv",
        index=False,
    )

    edge_df_g2[top_cols].head(20).to_csv(
        out_dir / "top_critical_lines_layer2.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Spearman correlation: attention vs loading_pct
    # ------------------------------------------------------------------
    correlation_rows = []

    for layer_name, df in [
        ("gat_layer_1", edge_df_g1),
        ("gat_layer_2", edge_df_g2),
    ]:
        valid = df.dropna(subset=["attn_mean", "loading_pct_mean"])

        if len(valid) >= 3:
            rho, p_value = _safe_spearman(
                valid["attn_mean"].values,
                valid["loading_pct_mean"].values,
            )
        else:
            rho, p_value = float("nan"), None

        correlation_rows.append(
            {
                "layer": layer_name,
                "n_edges": int(len(valid)),
                "spearman_rho_attention_vs_loading": rho,
                "p_value": p_value,
            }
        )

    corr_df = pd.DataFrame(correlation_rows)

    corr_df.to_csv(
        out_dir / "attention_loading_correlation.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Aggregate risk probabilities
    # ------------------------------------------------------------------
    mean_probs = np.mean(all_probs, axis=0)

    risk_df = pd.DataFrame(
        {
            "bus": np.arange(mean_probs.shape[0]),
            "fault_prob": mean_probs[:, 0],
            "congestion_prob": mean_probs[:, 1],
            "voltage_prob": mean_probs[:, 2],
        }
    )

    if bus_names_list:
        risk_df["bus_name"] = [
            bus_names_list[i] if i < len(bus_names_list) else str(i)
            for i in range(len(risk_df))
        ]

    risk_df["max_risk"] = risk_df[
        [
            "fault_prob",
            "congestion_prob",
            "voltage_prob",
        ]
    ].max(axis=1)

    risk_df["dominant_risk"] = risk_df[
        [
            "fault_prob",
            "congestion_prob",
            "voltage_prob",
        ]
    ].idxmax(axis=1).map(
        {
            "fault_prob": "Fault Risk",
            "congestion_prob": "Congestion Risk",
            "voltage_prob": "Voltage Instability",
        }
    )

    risk_df.sort_values("max_risk", ascending=False).head(20).to_csv(
        out_dir / "top_risk_buses.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    pos = ieee33_positions()

    # Fix:
    # build_graph uses add_reverse_edges=True, so attention contains both
    # directed pairs (u -> v) and (v -> u). For an undirected physical feeder
    # visualization, average both directions before building nx.Graph().
    edge_df_undirected = (
        edge_df_g1.copy()
        .assign(
            u=lambda d: d[["from_bus", "to_bus"]].min(axis=1).astype(int),
            v=lambda d: d[["from_bus", "to_bus"]].max(axis=1).astype(int),
        )
        .groupby(["u", "v"], as_index=False)
        .agg(attn_mean=("attn_mean", "mean"))
    )

    G = nx.Graph()

    for _, row in edge_df_undirected.iterrows():
        G.add_edge(
            int(row["u"]),
            int(row["v"]),
            weight=float(row["attn_mean"]),
        )

    nodes_in_pos = [n for n in G.nodes() if n in pos]
    sub_pos = {n: pos[n] for n in nodes_in_pos}
    edges_in_pos = [(u, v) for u, v in G.edges() if u in pos and v in pos]

    if len(edges_in_pos) > 0:
        weights = np.array(
            [
                G[u][v]["weight"]
                for u, v in edges_in_pos
            ]
        )

        w_min = float(weights.min())
        w_max = float(weights.max())

        widths = 1.0 + 5.0 * (weights - w_min) / (w_max - w_min + 1e-8)

    else:
        weights = np.array([0.0])
        w_min = 0.0
        w_max = 1.0
        widths = 1.0

    node_risk = {
        int(r["bus"]): float(r["max_risk"])
        for _, r in risk_df.iterrows()
    }

    node_colors = [
        node_risk.get(n, 0.0)
        for n in nodes_in_pos
    ]

    fig, ax = plt.subplots(figsize=(14, 6))

    nx.draw_networkx_edges(
        G,
        sub_pos,
        edgelist=edges_in_pos,
        width=widths,
        edge_color=weights,
        edge_cmap=plt.cm.Reds,
        alpha=0.85,
        ax=ax,
    )

    nx.draw_networkx_nodes(
        G,
        sub_pos,
        nodelist=nodes_in_pos,
        node_size=400,
        node_color=node_colors,
        cmap=plt.cm.Oranges,
        vmin=0.0,
        vmax=1.0,
        edgecolors="black",
        linewidths=0.8,
        ax=ax,
    )

    nx.draw_networkx_labels(
        G,
        sub_pos,
        font_size=7,
        ax=ax,
    )

    sm_e = plt.cm.ScalarMappable(
        cmap=plt.cm.Reds,
        norm=plt.Normalize(vmin=w_min, vmax=w_max),
    )
    sm_e.set_array([])

    fig.colorbar(
        sm_e,
        ax=ax,
        fraction=0.025,
        pad=0.01,
    ).set_label(
        "GAT Layer-1 Attention Score Mean",
        fontsize=8,
    )

    sm_n = plt.cm.ScalarMappable(
        cmap=plt.cm.Oranges,
        norm=plt.Normalize(vmin=0.0, vmax=1.0),
    )
    sm_n.set_array([])

    fig.colorbar(
        sm_n,
        ax=ax,
        fraction=0.025,
        pad=0.04,
    ).set_label(
        "Max Predicted Risk Mean",
        fontsize=8,
    )

    ax.set_title(
        "Attention-Based Risk Attribution — IEEE 33-Bus Radial Feeder\n"
        "Aggregated over test set with attention-loading validation",
        fontsize=10,
    )

    ax.axis("off")

    plt.tight_layout()

    plt.savefig(
        out_dir / "attention_feeder.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"\nSaved: {out_dir / 'attention_feeder.png'}")
    print(f"Saved: {out_dir / 'top_critical_lines.csv'}")
    print(f"Saved: {out_dir / 'top_critical_lines_layer2.csv'}")
    print(f"Saved: {out_dir / 'top_risk_buses.csv'}")
    print(f"Saved: {out_dir / 'edge_attention_loading_layer1.csv'}")
    print(f"Saved: {out_dir / 'edge_attention_loading_layer2.csv'}")
    print(f"Saved: {out_dir / 'attention_loading_correlation.csv'}")

    print("\nAttention-loading correlation:")
    print(corr_df.to_string(index=False))

    print("\nTop 10 critical lines with physical loading:")
    print(edge_df_g1[top_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()