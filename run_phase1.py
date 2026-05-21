#!/usr/bin/env python3

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import torch

from src.simulation.run_powerflow import run_power_flow
from src.data.build_graph import build_graph


def main():
    dss_file = PROJECT_ROOT / "data/raw/ieee33/IEEE33.dss"
    out_dir = PROJECT_ROOT / "data/processed/phase1"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not dss_file.exists():
        raise FileNotFoundError(f"DSS file not found: {dss_file}")

    results = run_power_flow(
        dss_file=dss_file,
        mode="snapshot",
        load_mult=1.0,
        save_dir=out_dir,
    )

    if not results["converged"]:
        raise RuntimeError("Power flow did not converge.")

    bus_df = results["bus_df"]
    line_df = results["line_df"]
    transformer_df = results["transformer_df"]
    summary = results["summary"]

    nx_graph, pyg_data = build_graph(
        bus_df,
        line_df,
        transformer_df,
        add_reverse_edges=True,
        save_path=out_dir / "graph",
    )

    metadata = {
        **summary,
        "nx_nodes": nx_graph.number_of_nodes(),
        "nx_edges": nx_graph.number_of_edges(),
        "pyg_nodes": int(pyg_data.num_nodes),
        "pyg_edges": int(pyg_data.num_edges),
        "node_feat_dim": int(pyg_data.x.shape[1]),
        "edge_feat_dim": int(pyg_data.edge_attr.shape[1]),
    }

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Phase 1 complete.")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()