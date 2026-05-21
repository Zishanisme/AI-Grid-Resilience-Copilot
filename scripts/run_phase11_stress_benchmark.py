#!/usr/bin/env python3
"""
scripts/run_phase11_stress_benchmark.py

Phase 11: Stress-Regime Performance Benchmark.

"""

from __future__ import annotations

import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

_venv_win = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
_venv_unix = PROJECT_ROOT / ".venv" / "bin" / "python"

PYTHON_EXE = (
    str(_venv_win)
    if _venv_win.exists()
    else str(_venv_unix)
    if _venv_unix.exists()
    else sys.executable
)

STRESS_LEVELS = [1.00, 1.10, 1.20, 1.30, 1.40]
SEEDS = [11, 22, 33]
DATASET_SEED = 42

MODELS = [
    "mlp",
    "gcn",
    "gat",
    "stgcn",
    "stgat_nogate",
    "resiligraph_stgat",
]

EPOCHS = 150
SEQ_LEN = 20
LR = "8e-4"
PATIENCE = 20
REQUESTED_TIMESTEPS = 500

SUMMARY_METRICS = [
    "macro_f1",
    "macro_auc",
    "macro_auprc",
    "macro_recall",
]

PHASE2_DIR = PROJECT_ROOT / "data" / "processed" / "phase2"

SAVE_DIR = PROJECT_ROOT / "results" / "phase11"
DATASET_ARCHIVE_DIR = SAVE_DIR / "datasets"
ORIGINAL_PHASE2_BACKUP = DATASET_ARCHIVE_DIR / "_original_phase2"

RUNS_CSV = SAVE_DIR / "stress_benchmark_runs.csv"
SUMMARY_CSV = SAVE_DIR / "stress_benchmark_summary.csv"
BEST_CSV = SAVE_DIR / "stress_best_by_level.csv"
META_CSV = SAVE_DIR / "stress_dataset_metadata.csv"
JSON_PATH = SAVE_DIR / "stress_benchmark.json"


def stress_tag(stress: float) -> str:
    return f"stress_{stress:.2f}".replace(".", "_")


def archived_dataset_dir(stress: float) -> Path:
    return DATASET_ARCHIVE_DIR / stress_tag(stress)


def run_key(stress: float, seed: int, model: str) -> tuple[float, int, str]:
    return (round(float(stress), 2), int(seed), str(model))


def backup_original_phase2() -> None:
    if PHASE2_DIR.exists() and not ORIGINAL_PHASE2_BACKUP.exists():
        print("[BACKUP] Preserving original Phase 2 dataset...")
        ORIGINAL_PHASE2_BACKUP.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(PHASE2_DIR, ORIGINAL_PHASE2_BACKUP)
        print(f"[BACKUP] Saved original Phase 2 data to {ORIGINAL_PHASE2_BACKUP}")


def restore_original_phase2() -> None:
    if ORIGINAL_PHASE2_BACKUP.exists():
        print("\n[RESTORE] Restoring original Phase 2 dataset...")
        if PHASE2_DIR.exists():
            shutil.rmtree(PHASE2_DIR)
        shutil.copytree(ORIGINAL_PHASE2_BACKUP, PHASE2_DIR)
        print(f"[RESTORE] Phase 2 data restored from {ORIGINAL_PHASE2_BACKUP}")
    else:
        print(
            "[WARN] No original Phase 2 backup found. "
            "Run Phase 2 again before Phase 4/6/7 if needed."
        )


def load_existing_runs() -> pd.DataFrame:
    if RUNS_CSV.exists():
        df = pd.read_csv(RUNS_CSV)
        print(f"[RESUME] Loaded existing runs: {len(df)} rows")
        return df

    return pd.DataFrame()


def completed_keys(existing_df: pd.DataFrame) -> set:
    if existing_df.empty:
        return set()

    required = {"stress_scale", "seed", "model"}

    if not required.issubset(existing_df.columns):
        return set()

    done = set()

    for _, row in existing_df.iterrows():
        if pd.isna(row.get("stress_scale")) or pd.isna(row.get("seed")):
            continue

        if pd.isna(row.get("macro_f1")) and pd.isna(row.get("macro_auprc")):
            continue

        done.add(
            run_key(
                float(row["stress_scale"]),
                int(row["seed"]),
                str(row["model"]),
            )
        )

    return done


def generate_dataset_for_stress(stress_scale: float) -> dict:
    print("\n" + "=" * 90)
    print(
        f"GENERATING DATASET | stress_scale={stress_scale:.2f} "
        f"| dataset_seed={DATASET_SEED}"
    )
    print("=" * 90)

    cmd = [
        PYTHON_EXE,
        "scripts/run_phase2_dataset.py",
        "--stress_scale",
        str(stress_scale),
        "--seed",
        str(DATASET_SEED),
    ]

    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)

    meta = read_dataset_metadata(PHASE2_DIR, stress_scale)

    archive_dir = archived_dataset_dir(stress_scale)
    archive_dir.parent.mkdir(parents=True, exist_ok=True)

    if archive_dir.exists():
        shutil.rmtree(archive_dir)

    shutil.copytree(PHASE2_DIR, archive_dir)

    return meta


def restore_dataset_for_stress(stress_scale: float) -> dict:
    archive_dir = archived_dataset_dir(stress_scale)

    if not archive_dir.exists():
        return generate_dataset_for_stress(stress_scale)

    print(f"[DATASET] Restoring archived dataset for stress={stress_scale:.2f}")

    if PHASE2_DIR.exists():
        shutil.rmtree(PHASE2_DIR)

    shutil.copytree(archive_dir, PHASE2_DIR)

    return read_dataset_metadata(PHASE2_DIR, stress_scale)


def ensure_dataset_for_stress(stress_scale: float) -> dict:
    archive_dir = archived_dataset_dir(stress_scale)

    if archive_dir.exists():
        return restore_dataset_for_stress(stress_scale)

    return generate_dataset_for_stress(stress_scale)


def read_dataset_metadata(data_dir: Path, stress_scale: float) -> dict:
    meta = {
        "stress_scale": float(stress_scale),
        "dataset_seed": int(DATASET_SEED),
        "n_total": np.nan,
        "n_train": np.nan,
        "n_val": np.nan,
        "n_test": np.nan,
        "valid_snapshot_count": np.nan,
        "requested_timesteps": REQUESTED_TIMESTEPS,
        "convergence_rate_pct": np.nan,
    }

    summary_path = data_dir / "phase2_metadata.json"

    if summary_path.exists():
        try:
            loaded = json.loads(summary_path.read_text())

            for key in [
                "n_total",
                "n_train",
                "n_val",
                "n_test",
                "stress_scale",
                "seed",
            ]:
                if key in loaded:
                    meta[key if key != "seed" else "dataset_seed"] = loaded[key]

            if "n_total" in loaded:
                n_total = int(loaded["n_total"])
                meta["valid_snapshot_count"] = n_total
                meta["convergence_rate_pct"] = round(
                    100.0 * n_total / REQUESTED_TIMESTEPS,
                    1,
                )

            return meta

        except Exception as e:
            print(f"[WARN] Could not read phase2_metadata.json: {e}")

    graph_path = data_dir / "temporal_graphs.pt"

    if graph_path.exists():
        try:
            graphs = torch.load(graph_path, weights_only=False)
            n_total = len(graphs)

            meta["n_total"] = n_total
            meta["valid_snapshot_count"] = n_total
            meta["convergence_rate_pct"] = round(
                100.0 * n_total / REQUESTED_TIMESTEPS,
                1,
            )

        except Exception as e:
            print(f"[WARN] Could not load temporal_graphs.pt: {e}")

    for split_name in ["train", "val", "test"]:
        path = data_dir / f"{split_name}_graphs.pt"

        if path.exists():
            try:
                split = torch.load(path, weights_only=False)
                meta[f"n_{split_name}"] = len(split)

            except Exception as e:
                print(f"[WARN] Could not load {split_name}_graphs.pt: {e}")

    return meta


def run_model(model_name: str, seed: int) -> dict:
    print(f"\nTRAINING | model={model_name} | model_seed={seed}")

    cmd = [
        PYTHON_EXE,
        "scripts/run_phase3_train.py",
        "--model",
        model_name,
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
        / model_name
        / "test_metrics.csv"
    )

    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    df = pd.read_csv(metrics_path)

    if df.empty:
        raise ValueError(f"Empty metrics file: {metrics_path}")

    return df.iloc[0].to_dict()


def summarize_runs(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame()

    rows = []

    for (stress, model), sub in results_df.groupby(["stress_scale", "model"]):
        row = {
            "stress_scale": float(stress),
            "model": model,
            "n_runs": int(len(sub)),
        }

        for metric in SUMMARY_METRICS:
            vals = [float(v) for v in sub[metric].dropna().tolist()]

            if len(vals) >= 2:
                row[f"{metric}_mean"] = round(statistics.mean(vals), 4)
                row[f"{metric}_std"] = round(statistics.stdev(vals), 4)
                row[f"{metric}_min"] = round(min(vals), 4)
                row[f"{metric}_max"] = round(max(vals), 4)
            elif len(vals) == 1:
                row[f"{metric}_mean"] = round(vals[0], 4)
                row[f"{metric}_std"] = 0.0
                row[f"{metric}_min"] = round(vals[0], 4)
                row[f"{metric}_max"] = round(vals[0], 4)
            else:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_std"] = np.nan
                row[f"{metric}_min"] = np.nan
                row[f"{metric}_max"] = np.nan

        for key in [
            "n_total",
            "n_train",
            "n_val",
            "n_test",
            "valid_snapshot_count",
            "requested_timesteps",
            "convergence_rate_pct",
            "dataset_seed",
        ]:
            if key in sub.columns:
                clean = sub[key].dropna()
                row[key] = clean.iloc[0] if not clean.empty else np.nan
            else:
                row[key] = np.nan

        rows.append(row)

    summary_df = pd.DataFrame(rows)

    summary_df = summary_df.sort_values(
        ["stress_scale", "macro_auprc_mean"],
        ascending=[True, False],
    )

    return summary_df


def best_by_stress(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    rows = []

    for stress, sub in summary_df.groupby("stress_scale"):
        row = {"stress_scale": float(stress)}

        for metric in SUMMARY_METRICS:
            col = f"{metric}_mean"
            valid = sub.dropna(subset=[col])

            if valid.empty:
                row[f"best_{metric}_model"] = None
                row[f"best_{metric}"] = np.nan
            else:
                best = valid.loc[valid[col].idxmax()]
                row[f"best_{metric}_model"] = best["model"]
                row[f"best_{metric}"] = best[col]

        rows.append(row)

    return pd.DataFrame(rows)


def save_all(results_df: pd.DataFrame, meta_df: pd.DataFrame) -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    if not results_df.empty:
        results_df = results_df.drop_duplicates(
            subset=["stress_scale", "seed", "model"],
            keep="last",
        ).sort_values(["stress_scale", "seed", "model"])

        results_df.to_csv(RUNS_CSV, index=False)

    summary_df = summarize_runs(results_df)
    best_df = best_by_stress(summary_df)

    summary_df.to_csv(SUMMARY_CSV, index=False)
    best_df.to_csv(BEST_CSV, index=False)

    if not meta_df.empty:
        meta_df = meta_df.drop_duplicates(
            subset=["stress_scale"],
            keep="last",
        ).sort_values(["stress_scale"])

        meta_df.to_csv(META_CSV, index=False)

    with open(JSON_PATH, "w") as f:
        json.dump(
            {
                "stress_levels": STRESS_LEVELS,
                "model_seeds": SEEDS,
                "dataset_seed": DATASET_SEED,
                "models": MODELS,
                "epochs": EPOCHS,
                "patience": PATIENCE,
                "runs": (
                    results_df.to_dict(orient="records")
                    if not results_df.empty
                    else []
                ),
                "summary": summary_df.to_dict(orient="records"),
                "best_by_stress": best_df.to_dict(orient="records"),
                "dataset_metadata": (
                    meta_df.to_dict(orient="records") if not meta_df.empty else []
                ),
                "framing": (
                    "Stress-regime performance benchmark. Each stress level "
                    "uses one fixed generated dataset and evaluates model "
                    "initialization variability across three seeds. This is not "
                    "out-of-distribution robustness."
                ),
                "phase2_restore_note": (
                    "Original Phase 2 dataset is backed up before Phase 11 and "
                    "restored after Phase 11 completes."
                ),
            },
            f,
            indent=2,
            default=str,
        )


def print_progress(results_df: pd.DataFrame) -> None:
    total = len(STRESS_LEVELS) * len(SEEDS) * len(MODELS)

    if results_df.empty:
        done = 0
    else:
        done = len(
            results_df.drop_duplicates(
                subset=["stress_scale", "seed", "model"],
                keep="last",
            )
        )

    print("\n" + "-" * 90)
    print(f"PROGRESS: {done}/{total} runs saved")
    print(f"Saved: {RUNS_CSV}")
    print("-" * 90)


def print_final_summary(results_df: pd.DataFrame) -> None:
    summary_df = summarize_runs(results_df)
    best_df = best_by_stress(summary_df)

    print("\n" + "=" * 90)
    print("PHASE 11 SUMMARY — STRESS-REGIME PERFORMANCE")
    print("=" * 90)

    if summary_df.empty:
        print("No completed runs.")
        return

    display_cols = [
        "stress_scale",
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
        "valid_snapshot_count",
        "convergence_rate_pct",
    ]

    existing = [c for c in display_cols if c in summary_df.columns]
    print(summary_df[existing].to_string(index=False))

    print("\n" + "=" * 90)
    print("BEST MODEL BY STRESS LEVEL")
    print("=" * 90)
    print(best_df.to_string(index=False))


def main() -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    backup_original_phase2()

    results_df = load_existing_runs()
    meta_df = pd.read_csv(META_CSV) if META_CSV.exists() else pd.DataFrame()

    done = completed_keys(results_df)

    try:
        for stress in STRESS_LEVELS:
            dataset_meta = ensure_dataset_for_stress(stress)

            meta_df = pd.concat(
                [meta_df, pd.DataFrame([dataset_meta])],
                ignore_index=True,
            )

            save_all(results_df=results_df, meta_df=meta_df)

            for seed in SEEDS:
                for model in MODELS:
                    key = run_key(stress, seed, model)

                    if key in done:
                        print(
                            f"[SKIP] Already completed | "
                            f"stress={stress:.2f} seed={seed} model={model}"
                        )
                        continue

                    try:
                        metrics = run_model(model_name=model, seed=seed)

                        row = {
                            "stress_scale": float(stress),
                            "seed": int(seed),
                            "dataset_seed": int(DATASET_SEED),
                            "model": model,
                            "macro_f1": metrics.get("macro_f1", np.nan),
                            "macro_auc": metrics.get("macro_auc", np.nan),
                            "macro_auprc": metrics.get("macro_auprc", np.nan),
                            "macro_recall": metrics.get("macro_recall", np.nan),
                        }

                        for meta_key in [
                            "n_total",
                            "n_train",
                            "n_val",
                            "n_test",
                            "valid_snapshot_count",
                            "requested_timesteps",
                            "convergence_rate_pct",
                        ]:
                            row[meta_key] = dataset_meta.get(meta_key, np.nan)

                        results_df = pd.concat(
                            [results_df, pd.DataFrame([row])],
                            ignore_index=True,
                        )

                        done.add(key)

                        save_all(results_df=results_df, meta_df=meta_df)
                        print_progress(results_df)

                        print(
                            f"COMPLETED | stress={stress:.2f} seed={seed} "
                            f"{model:<22} "
                            f"F1={float(row['macro_f1']):.4f} "
                            f"AUC={float(row['macro_auc']):.4f} "
                            f"AUPRC={float(row['macro_auprc']):.4f} "
                            f"Recall={float(row['macro_recall']):.4f}"
                        )

                    except Exception as e:
                        print(
                            f"[WARN] Failed | stress={stress:.2f} "
                            f"seed={seed} model={model}: {e}"
                        )

                        row = {
                            "stress_scale": float(stress),
                            "seed": int(seed),
                            "dataset_seed": int(DATASET_SEED),
                            "model": model,
                            "macro_f1": np.nan,
                            "macro_auc": np.nan,
                            "macro_auprc": np.nan,
                            "macro_recall": np.nan,
                        }

                        for meta_key in [
                            "n_total",
                            "n_train",
                            "n_val",
                            "n_test",
                            "valid_snapshot_count",
                            "requested_timesteps",
                            "convergence_rate_pct",
                        ]:
                            row[meta_key] = dataset_meta.get(meta_key, np.nan)

                        results_df = pd.concat(
                            [results_df, pd.DataFrame([row])],
                            ignore_index=True,
                        )

                        save_all(results_df=results_df, meta_df=meta_df)
                        print_progress(results_df)

        save_all(results_df=results_df, meta_df=meta_df)
        print_final_summary(results_df)

        print(f"\nSaved runs CSV      → {RUNS_CSV}")
        print(f"Saved summary CSV   → {SUMMARY_CSV}")
        print(f"Saved best CSV      → {BEST_CSV}")
        print(f"Saved metadata CSV  → {META_CSV}")
        print(f"Saved JSON          → {JSON_PATH}")

    finally:
        restore_original_phase2()


if __name__ == "__main__":
    main()