#!/usr/bin/env python3

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import torch
import numpy as np

from src.data.temporal_dataset import build_temporal_graph_dataset
from src.data.normalization import fit_feature_stats, apply_feature_stats
from src.data.splits import temporal_split


def build_load_profile(n_steps=500, seed=42, stress_scale=1.0):
    np.random.seed(seed)

    base = (
        1.45
        + (0.95 * stress_scale)
        * np.sin(np.linspace(0, 20 * np.pi, n_steps))
    )

    noise = np.random.normal(
        0,
        0.15 * stress_scale,
        size=n_steps,
    )

    spikes = np.random.choice(
        [
            0.0,
            0.65 * stress_scale,
            1.00 * stress_scale,
        ],
        size=n_steps,
        p=[0.68, 0.22, 0.10],
    )

    profile = base + noise + spikes

    return np.clip(
        profile,
        0.55,
        3.20 * stress_scale,
    )


def main(stress_scale=1.0, seed=42):
    dss_file = PROJECT_ROOT / "data/raw/ieee33/IEEE33.dss"
    out_dir = PROJECT_ROOT / "data/processed/phase2"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not dss_file.exists():
        raise FileNotFoundError(f"DSS file not found: {dss_file}")

    load_profile = build_load_profile(
        n_steps=500,
        seed=seed,
        stress_scale=stress_scale,
    )

    graphs = build_temporal_graph_dataset(
        dss_file=dss_file,
        load_profile=load_profile,
        out_dir=out_dir,
        renewable_aware=True,
        seed=seed,
    )

    train, val, test = temporal_split(graphs)

    stats = fit_feature_stats(train)

    train = apply_feature_stats(train, stats)
    val = apply_feature_stats(val, stats)
    test = apply_feature_stats(test, stats)

    torch.save(train, out_dir / "train_graphs.pt")
    torch.save(val, out_dir / "val_graphs.pt")
    torch.save(test, out_dir / "test_graphs.pt")
    torch.save(stats, out_dir / "feature_stats.pt")

    metadata = {
        "n_total": len(graphs),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "tasks": [
            "fault_risk",
            "congestion_risk",
            "voltage_violation",
        ],
        "split": "temporal",
        "load_profile": (
            "controlled semi-stressed renewable-aware 500-step sinusoidal "
            "profile with Gaussian noise and stress-scaled demand spikes"
        ),
        "stress_scale": stress_scale,
        "seed": seed,
    }

    with open(out_dir / "phase2_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Phase 2 complete.")
    print(metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stress_scale",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    main(
        stress_scale=args.stress_scale,
        seed=args.seed,
    )