#!/usr/bin/env python3
"""
scripts/run_phase3_train.py — Phase 3: train one or all models on Phase 2 data.

Usage
-----
    python scripts/run_phase3_train.py
    python scripts/run_phase3_train.py --model resiligraph_stgat --seed 42
    python scripts/run_phase3_train.py --model stgat_nogate --seed 42
    python scripts/run_phase3_train.py --epochs 400 --seq_len 20 --seed 11

Fixes:
- controlled --seed for reproducibility,
- added STGAT_NoGate ablation,
- saves n_params for reviewer-safe comparison.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.baselines import MLPBaseline, GCNBaseline, GATBaseline
from src.models.st_gnn import (
    STGAT_GRU,
    STGCN_GRU,
    STGAT_NoGate,
)
from src.train.train_model import train_model
from src.train.utils import check_label_balance


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "phase3_run.log", mode="a"),
    ],
)

log = logging.getLogger("phase3")


def set_seed(seed: int) -> None:
    """Fix all random sources for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    log.info("Seed set to %d", seed)


def load_splits():
    data_dir = PROJECT_ROOT / "data/processed/phase2"

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Phase 2 data not found at {data_dir}. "
            "Run scripts/run_phase2_dataset.py first."
        )

    train = torch.load(data_dir / "train_graphs.pt", weights_only=False)
    val = torch.load(data_dir / "val_graphs.pt", weights_only=False)
    test = torch.load(data_dir / "test_graphs.pt", weights_only=False)

    log.info(
        "Loaded: train=%d  val=%d  test=%d",
        len(train),
        len(val),
        len(test),
    )

    return train, val, test


def build_models(in_dim: int) -> dict:
    return {
        "mlp": MLPBaseline(in_dim),
        "gcn": GCNBaseline(in_dim),
        "gat": GATBaseline(in_dim),
        "stgcn": STGCN_GRU(in_dim),
        "stgat_nogate": STGAT_NoGate(in_dim),
        "resiligraph_stgat": STGAT_GRU(in_dim),
    }


def main(args) -> int:
    set_seed(args.seed)

    train, val, test = load_splits()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_dim = train[0].x.shape[1]

    log.info(
        "Device=%s | in_dim=%d | seed=%d",
        device,
        in_dim,
        args.seed,
    )

    check_label_balance(train, val, test)

    all_models = build_models(in_dim)

    if args.model and args.model != "all":
        if args.model not in all_models:
            log.error(
                "Unknown model %r. Choices: %s",
                args.model,
                list(all_models),
            )
            return 1

        models_to_run = {args.model: all_models[args.model]}

    else:
        models_to_run = all_models

    results = {}

    for name, model in models_to_run.items():
        n_params = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

        log.info(
            "\n%s\n  MODEL: %s  (params=%d)  seed=%d\n%s",
            "=" * 60,
            name.upper(),
            n_params,
            args.seed,
            "=" * 60,
        )

        out_dir = PROJECT_ROOT / "results" / "phase3" / name

        metrics = train_model(
            model=model,
            train_graphs=train,
            val_graphs=val,
            test_graphs=test,
            out_dir=out_dir,
            device=device,
            seq_len=args.seq_len,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
        )

        results[name] = {
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in metrics.items()
        }

        results[name]["n_params"] = n_params
        results[name]["seed"] = args.seed

        pred_src = out_dir / "test_predictions.csv"

        if pred_src.exists() and name == "resiligraph_stgat":
            log.info("test_predictions.csv available at %s", pred_src)

    out = PROJECT_ROOT / "results" / "phase3" / "model_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w") as fh:
        json.dump(results, fh, indent=2, default=str)

    log.info("Comparison saved → %s", out)

    _print_table(results)

    return 0


def _print_table(results: dict) -> None:
    metrics = [
        "macro_f1",
        "macro_auc",
        "macro_auprc",
        "macro_recall",
    ]

    w = 14

    hdr = f"{'Model':<25}" + "".join(
        f"{m:>{w}}"
        for m in metrics
    )

    print(f"\n{'─' * len(hdr)}\n{hdr}\n{'─' * len(hdr)}")

    for name, m in results.items():
        row = f"{name:<25}"

        for met in metrics:
            v = m.get(met)

            row += (
                f"{v:>{w}.4f}"
                if isinstance(v, float)
                else f"{'N/A':>{w}}"
            )

        print(row)

    print(f"{'─' * len(hdr)}\n")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--model",
        default="all",
        help="Model key or 'all'",
    )

    p.add_argument(
        "--epochs",
        type=int,
        default=400,
    )

    p.add_argument(
        "--seq_len",
        type=int,
        default=20,
        help="Sequence length — must match Phase 4 and Phase 7",
    )

    p.add_argument(
        "--lr",
        type=float,
        default=8e-4,
    )

    p.add_argument(
        "--patience",
        type=int,
        default=60,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    return p.parse_args()


if __name__ == "__main__":
    sys.exit(main(_parse()))