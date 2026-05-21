"""
evaluate.py
===========
Model evaluation utilities.

Fix (R5): tune_thresholds now guards both the all-negative case
(yt.sum() == 0) and the all-positive case (yt.sum() == len(yt)).
Both fall back to 0.5 — threshold search is meaningless when only
one class is present in the validation split.
"""

import torch
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
)

TASKS = ["fault_risk", "congestion_risk", "voltage_violation"]


@torch.no_grad()
def collect_predictions(model, sequences, device):
    model.eval()
    y_true_all, y_prob_all, all_rows = [], [], []

    for seq_idx, seq in enumerate(sequences):
        seq    = [g.to(device) for g in seq]
        logits = model(seq)
        probs  = torch.sigmoid(logits)
        probs  = torch.clamp(probs, 1e-6, 1.0 - 1e-6)

        y_true_all.append(seq[-1].y.detach().cpu())
        y_prob_all.append(probs.detach().cpu())

        probs_np = probs.detach().cpu().numpy()
        timestep = getattr(seq[-1], "timestep", seq_idx)

        for i in range(probs_np.shape[0]):
            all_rows.append({
                "timestep":        int(timestep),
                "bus":             int(i),
                "fault_prob":      float(probs_np[i, 0]),
                "congestion_prob": float(probs_np[i, 1]),
                "voltage_prob":    float(probs_np[i, 2]),
            })

    if not y_true_all:
        return None, None, pd.DataFrame()

    y_true  = torch.cat(y_true_all, dim=0).numpy()
    y_prob  = torch.cat(y_prob_all, dim=0).numpy()
    pred_df = pd.DataFrame(all_rows)
    return y_true, y_prob, pred_df


def tune_thresholds(y_true, y_prob, thresholds=None):
    """
    Per-task threshold tuning on validation F1.

    R5 fix: degenerate guard covers both all-negative AND all-positive tasks.
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 37)

    best_thresholds = []
    for i, task in enumerate(TASKS):
        yt = y_true[:, i]
        yp = y_prob[:, i]

        # R5 fix: skip search when only one class present
        if yt.sum() == 0 or yt.sum() == len(yt):
            best_thresholds.append(0.5)
            continue

        best_t, best_f1 = 0.5, -1.0
        for t in thresholds:
            f1 = f1_score(yt, (yp >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)

        best_thresholds.append(best_t)

    return np.array(best_thresholds, dtype=float)


def compute_metrics(y_true, y_prob, thresholds=None, verbose=False):
    if thresholds is None:
        thresholds = np.array([0.5, 0.5, 0.5], dtype=float)

    thresholds = np.asarray(thresholds, dtype=float)
    y_pred     = np.zeros_like(y_prob, dtype=int)
    for i in range(len(TASKS)):
        y_pred[:, i] = (y_prob[:, i] >= thresholds[i]).astype(int)

    metrics: dict = {}
    for i, task in enumerate(TASKS):
        yt, yp, yh = y_true[:, i], y_prob[:, i], y_pred[:, i]
        if verbose:
            print(f"{task}: pos={int(yt.sum())}, total={len(yt)}, thr={thresholds[i]:.3f}")

        metrics[f"{task}_threshold"] = float(thresholds[i])
        metrics[f"{task}_f1"]        = f1_score(yt, yh, zero_division=0)
        metrics[f"{task}_precision"] = precision_score(yt, yh, zero_division=0)
        metrics[f"{task}_recall"]    = recall_score(yt, yh, zero_division=0)
        try:    metrics[f"{task}_auc"]   = roc_auc_score(yt, yp)
        except: metrics[f"{task}_auc"]   = 0.5
        try:    metrics[f"{task}_auprc"] = average_precision_score(yt, yp)
        except: metrics[f"{task}_auprc"] = 0.0

    for agg, key in [("macro_f1","_f1"), ("macro_precision","_precision"),
                     ("macro_recall","_recall"), ("macro_auc","_auc"),
                     ("macro_auprc","_auprc")]:
        metrics[agg] = float(np.mean([metrics[f"{t}{key}"] for t in TASKS]))

    return metrics


@torch.no_grad()
def evaluate(model, sequences, device, threshold=0.5,
             thresholds=None, verbose=False, return_predictions=False):
    y_true, y_prob, pred_df = collect_predictions(model, sequences, device)

    if y_true is None:
        empty = {"macro_f1": 0.0, "macro_precision": 0.0, "macro_recall": 0.0,
                 "macro_auc": 0.5, "macro_auprc": 0.0}
        return (empty, pred_df) if return_predictions else empty

    if thresholds is None:
        thresholds = np.array([threshold, threshold, threshold], dtype=float)

    metrics = compute_metrics(y_true=y_true, y_prob=y_prob,
                              thresholds=thresholds, verbose=verbose)
    return (metrics, pred_df) if return_predictions else metrics


def print_metrics(metrics: dict, split: str = "") -> None:
    if split:
        print(f"\n=== {split.upper()} METRICS ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}" if value is not None else f"{key}: None")
