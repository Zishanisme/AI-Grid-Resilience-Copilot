#!/usr/bin/env python3
"""
scripts/run_phase5_renewable.py — Phase 5: Fair Renewable-Aware Ablation.

Fix (M5 / W8): Masking now occurs BEFORE normalization statistics are fit.
Each configuration (masked / full) has its own normalization stats fitted on
its own training-set distribution.  This ensures:
  - Masked columns (zeroed) normalize to exactly 0.0 after z-score transform.
  - Non-renewable channels are normalized within their own feature space.

Previously, masking happened after shared normalization, so zeroed channels
received (0 − mean_i) / std_i — a constant negative value, not a true
zero-information signal.

All other ablation fairness invariants are preserved:
  - Same renewable-risk dataset (renewable_aware=True generation).
  - Same labels and same temporal split.
  - Same model architecture and hyperparameters for both configs.
  - Only the information available to the model differs.
"""

from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.temporal_dataset import build_temporal_graph_dataset
from src.data.normalization    import fit_feature_stats, apply_feature_stats
from src.data.splits           import temporal_split
from src.models.st_gnn         import STGAT_GRU
from src.train.train_model     import train_model


def build_load_profile(n_steps: int = 500, seed: int = 42) -> np.ndarray:
    np.random.seed(seed)
    base   = 1.0 + 0.5 * np.sin(np.linspace(0, 20 * np.pi, n_steps))
    noise  = np.random.normal(0, 0.08, size=len(base))
    spikes = np.random.choice([0, 0.35], size=len(base), p=[0.88, 0.12])
    return np.clip(base + noise + spikes, 0.4, 1.8)


def mask_renewable_features(graphs: list) -> list:
    """
    Zero out renewable-aware feature channels.

    Feature index map (9-dim renewable-aware vector):
        0  load_pu           — retained
        1  solar_pu          — zeroed
        2  net_load_pu       — replaced with load_pu (most informative proxy)
        3  renewable_ramp    — zeroed
        4  hour_sin          — retained
        5  hour_cos          — retained
        6  voltage_dev       — retained
        7  degree_norm       — retained
        8  renewable_stress  — zeroed

    Returns a deepcopy so the original graphs are not mutated.
    """
    masked = copy.deepcopy(graphs)
    for g in masked:
        g.x[:, 1] = 0.0         # remove solar
        g.x[:, 2] = g.x[:, 0]  # net_load → raw load (non-renewable proxy)
        g.x[:, 3] = 0.0         # remove renewable ramp
        g.x[:, 8] = 0.0         # remove renewable stress
    return masked


def prepare_raw_split(dss_file: Path, out_dir: Path):
    """
    Generate the shared renewable-aware dataset and split it.
    Returns UN-normalized (raw) train / val / test lists.
    Raw splits are saved so they can be independently normalized per config.
    """
    load_profile = build_load_profile()

    graphs = build_temporal_graph_dataset(
        dss_file=dss_file,
        load_profile=load_profile,
        out_dir=out_dir,
        renewable_aware=True,
        seed=42,
    )

    train, val, test = temporal_split(graphs)

    # Save raw splits (pre-normalization) for reproducibility
    torch.save(train, out_dir / "raw_train.pt")
    torch.save(val,   out_dir / "raw_val.pt")
    torch.save(test,  out_dir / "raw_test.pt")

    return train, val, test


def normalize_config(
    train_raw: list,
    val_raw:   list,
    test_raw:  list,
    mask:      bool,
    stats_path: Path,
) -> tuple:
    """
    Apply masking FIRST, then fit normalization stats on masked train set.

    This ensures the masked model sees exactly 0.0 for removed channels
    (after z-score: (0 - 0) / 1e-6 = 0.0) rather than a constant
    derived from the full-feature distribution.
    """
    if mask:
        train = mask_renewable_features(train_raw)
        val   = mask_renewable_features(val_raw)
        test  = mask_renewable_features(test_raw)
    else:
        train = copy.deepcopy(train_raw)
        val   = copy.deepcopy(val_raw)
        test  = copy.deepcopy(test_raw)

    # Fit stats on THIS config's training set
    stats = fit_feature_stats(train)
    torch.save(stats, stats_path)

    train = apply_feature_stats(train, stats)
    val   = apply_feature_stats(val,   stats)
    test  = apply_feature_stats(test,  stats)

    return train, val, test


def train_one_config(
    name:      str,
    train:     list,
    val:       list,
    test:      list,
    mask:      bool,
) -> dict:
    print("\n" + "=" * 70)
    print(f"Phase 5 config: {name}  |  mask_features={mask}")
    print("=" * 70)

    result_dir = PROJECT_ROOT / "results/phase5_renewable" / name

    if result_dir.exists():
        shutil.rmtree(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    in_dim = train[0].x.shape[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Feature dimension : {in_dim}")
    print(f"Masked renewable  : {mask}")

    model = STGAT_GRU(in_dim)

    metrics = train_model(
        model=model,
        train_graphs=train,
        val_graphs=val,
        test_graphs=test,
        out_dir=result_dir,
        device=device,
        seq_len=10,          # H2 fix: consistent with Phase 3
        epochs=300,
        lr=3e-4,
        patience=30,
        save_predictions=False,
    )

    return {
        "Configuration":   name,
        "Fault AUC":       metrics.get("fault_risk_auc"),
        "Congestion AUC":  metrics.get("congestion_risk_auc"),
        "Voltage AUC":     metrics.get("voltage_violation_auc"),
        "Mean AUC":        metrics.get("macro_auc"),
        "Fault F1":        metrics.get("fault_risk_f1"),
        "Congestion F1":   metrics.get("congestion_risk_f1"),
        "Voltage F1":      metrics.get("voltage_violation_f1"),
        "Mean F1":         metrics.get("macro_f1"),
        "Mean AUPRC":      metrics.get("macro_auprc"),
    }


def main():
    print("=" * 70)
    print("Phase 5 — Fair Renewable-Aware Ablation")
    print("=" * 70)

    dss_file = PROJECT_ROOT / "data/raw/ieee33/IEEE33.dss"
    out_root = PROJECT_ROOT / "results/phase5_renewable"
    data_dir = PROJECT_ROOT / "data/processed/phase5_shared_renewable"

    out_root.mkdir(parents=True, exist_ok=True)

    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1: Generate ONE dataset, split into raw (un-normalized) sets
    # -----------------------------------------------------------------------
    print("\nGenerating shared renewable-aware dataset...")
    train_raw, val_raw, test_raw = prepare_raw_split(dss_file, data_dir)

    # -----------------------------------------------------------------------
    # Step 2: Normalize each config independently after masking (M5 fix)
    # -----------------------------------------------------------------------
    print("\nNormalizing Config A (masked — no renewable features)...")
    train_a, val_a, test_a = normalize_config(
        train_raw, val_raw, test_raw,
        mask=True,
        stats_path=data_dir / "feature_stats_masked.pt",
    )

    print("\nNormalizing Config B (full — all renewable features)...")
    train_b, val_b, test_b = normalize_config(
        train_raw, val_raw, test_raw,
        mask=False,
        stats_path=data_dir / "feature_stats_full.pt",
    )

    # -----------------------------------------------------------------------
    # Step 3: Train both configs
    # -----------------------------------------------------------------------
    rows = [
        train_one_config(
            name="Without renewable feature channels",
            train=train_a, val=val_a, test=test_a,
            mask=True,
        ),
        train_one_config(
            name="With solar/net-load feature channels",
            train=train_b, val=val_b, test=test_b,
            mask=False,
        ),
    ]

    df = pd.DataFrame(rows)

    csv_path  = out_root / "renewable_comparison.csv"
    json_path = out_root / "renewable_comparison.json"

    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    print("\nTable 2 — Fair Renewable Ablation (mask-before-normalize)")
    print(df.to_string(index=False))
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
