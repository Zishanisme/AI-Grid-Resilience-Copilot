"""
train_model.py
==============
Training loop with early stopping, focal loss, threshold calibration,
and full metric logging.

Fix (R4): saves model_meta.json alongside best_model.pt so Phase 7
(run_phase7_rl.py) can reconstruct the model architecture without
fragile weight-shape inference.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data

from src.train.evaluate  import evaluate, print_metrics, collect_predictions, tune_thresholds
from src.train.losses    import make_loss
from src.train.utils     import make_sequences, save_checkpoint, load_checkpoint

log = logging.getLogger(__name__)


def _save_model_meta(model: nn.Module, out_dir: Path, train_graphs) -> None:
    """
    Save model architecture metadata for reproducible loading in Phase 7.

    R4 fix: without this file, run_phase7_rl.py must infer architecture
    from weight-tensor shapes, which breaks when PyG updates internal
    key names across versions.
    """
    in_dim     = train_graphs[0].x.shape[1]
    hidden_dim = int(getattr(model, "hidden_dim", 64))

    # GATConv stores heads as an attribute in PyG
    gat1 = getattr(model, "gat1", None)
    heads = int(getattr(gat1, "heads", 2)) if gat1 is not None else 2

    meta = {
        "in_dim":     in_dim,
        "hidden_dim": hidden_dim,
        "heads":      heads,
    }
    with open(out_dir / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def train_model(
    model:        nn.Module,
    train_graphs: List[Data],
    val_graphs:   List[Data],
    test_graphs:  List[Data],
    out_dir:      str | Path,
    *,
    device:       str | torch.device = "cpu",
    seq_len:      int   = 20,
    epochs:       int   = 100,
    lr:           float = 1e-3,
    weight_decay: float = 1e-4,
    patience:     int   = 15,
) -> Dict:
    """
    Train a model and return calibrated test-set metrics.

    Design
    ------
    - Weighted focal loss for rare-event grid-risk prediction.
    - Early stopping on val macro AUPRC (better than F1 for sparse labels).
    - ReduceLROnPlateau scheduler on val score.
    - Threshold calibration from validation set.
    - test_predictions.csv always saved (consumed by Phase 6 copilot).
    - model_meta.json saved for Phase 7 architecture loading (R4 fix).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # R4 fix: save architecture metadata immediately
    _save_model_meta(model, out_dir, train_graphs)

    train_seq = make_sequences(train_graphs, seq_len)
    val_seq   = make_sequences(val_graphs,   seq_len)
    test_seq  = make_sequences(test_graphs,  seq_len)

    if not train_seq:
        raise RuntimeError(
            f"No training sequences (train={len(train_graphs)}, seq_len={seq_len})."
        )

    log.info("Sequences — train:%d  val:%d  test:%d",
             len(train_seq), len(val_seq), len(test_seq))

    model     = model.to(device)
    criterion = make_loss(train_graphs, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=max(3, patience // 3),
    )

    best_score = -1.0
    bad_epochs = 0
    history: List[Dict] = []
    ckpt_path = out_dir / "best_model.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for seq in train_seq:
            seq_dev = [g.to(device) for g in seq]
            optimizer.zero_grad()
            logits = model(seq_dev)
            target = seq_dev[-1].y
            loss   = criterion(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item())

        train_loss  = total_loss / max(1, len(train_seq))
        val_metrics = evaluate(model, val_seq, device) if val_seq else {}
        val_score   = (
            val_metrics.get("macro_auprc")
            or val_metrics.get("macro_f1")
            or 0.0
        )
        scheduler.step(val_score)

        row: Dict = {
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "lr":         optimizer.param_groups[0]["lr"],
        }
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(row)

        log.info(
            "Epoch %03d | loss=%.4f | val_f1=%.4f | val_auprc=%.4f | val_auc=%.4f",
            epoch, train_loss,
            val_metrics.get("macro_f1",    0.0),
            val_metrics.get("macro_auprc", 0.0),
            val_metrics.get("macro_auc",   0.5),
        )

        if val_score > best_score:
            best_score = val_score
            bad_epochs = 0
            save_checkpoint(model, ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                log.info("Early stopping at epoch %d.", epoch)
                break

    if ckpt_path.exists():
        model = load_checkpoint(model, ckpt_path, device)

    # ------------------------------------------------------------------
    # Threshold calibration on validation set
    # ------------------------------------------------------------------
    if val_seq:
        y_val, p_val, _ = collect_predictions(model, val_seq, device)
        thresholds = tune_thresholds(y_val, p_val)
    else:
        thresholds = torch.tensor([0.5, 0.5, 0.5]).numpy()

    threshold_dict = {
        "fault_risk":        float(thresholds[0]),
        "congestion_risk":   float(thresholds[1]),
        "voltage_violation": float(thresholds[2]),
    }
    with open(out_dir / "thresholds.json", "w") as f:
        json.dump(threshold_dict, f, indent=2)
    print("\nCalibrated thresholds:", threshold_dict)

    # ------------------------------------------------------------------
    # Final test evaluation
    # ------------------------------------------------------------------
    if test_seq:
        test_metrics, pred_df = evaluate(
            model, test_seq, device,
            thresholds=thresholds, verbose=True, return_predictions=True,
        )
    else:
        test_metrics, pred_df = {}, pd.DataFrame()

    print_metrics(test_metrics, split="test")

    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    pd.DataFrame([test_metrics]).to_csv(out_dir / "test_metrics.csv", index=False)

    if len(pred_df) > 0:
        pred_df.to_csv(out_dir / "test_predictions.csv", index=False)

    return test_metrics
