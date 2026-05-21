import pickle
import torch
import pandas as pd
import networkx as nx
from pathlib import Path
from torch_geometric.data import Data

NODE_FEAT_COLS = [
    "q_load_kvar",
]

EDGE_FEAT_COLS = [
    "r_ohm",
    "x_ohm",
    "length",
    "loading_pct",
]


def _safe_col(df, col, default=0.0):
    if col not in df.columns:
        df[col] = default
    return df


def build_graph(bus_df, line_df, transformer_df=None, add_reverse_edges=True, save_path=None):
    bus_df = bus_df.copy()
    line_df = line_df.copy()

    for col in NODE_FEAT_COLS:
        bus_df = _safe_col(bus_df, col, 0.0)

    for col in EDGE_FEAT_COLS:
        line_df = _safe_col(line_df, col, 0.0)

    if "bus_name" not in bus_df.columns:
        raise ValueError("bus_df must contain bus_name")

    if "from_bus" not in line_df.columns or "to_bus" not in line_df.columns:
        raise ValueError("line_df must contain from_bus and to_bus")

    bus_names = list(bus_df["bus_name"].astype(str))
    bus_to_idx = {b: i for i, b in enumerate(bus_names)}

    G = nx.DiGraph()

    for _, row in bus_df.iterrows():
        b = str(row["bus_name"])
        G.add_node(
            b,
            **{c: float(row[c]) for c in NODE_FEAT_COLS},
        )

    edge_index = []
    edge_attr = []

    def add_edge(u, v, attrs):
        if u not in bus_to_idx or v not in bus_to_idx:
            return

        G.add_edge(u, v, **attrs)
        edge_index.append([bus_to_idx[u], bus_to_idx[v]])
        edge_attr.append([float(attrs.get(c, 0.0)) for c in EDGE_FEAT_COLS])

    for _, row in line_df.iterrows():
        u = str(row["from_bus"])
        v = str(row["to_bus"])

        attrs = {c: float(row[c]) for c in EDGE_FEAT_COLS}

        add_edge(u, v, attrs)

        if add_reverse_edges:
            add_edge(v, u, attrs)

    x = torch.tensor(
        bus_df[NODE_FEAT_COLS].astype(float).values,
        dtype=torch.float,
    )

    if len(edge_index) == 0:
        raise ValueError("No valid edges found while building graph")

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    y = torch.zeros((len(bus_df), 3), dtype=torch.float)

    pyg_data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
    )

    pyg_data.bus_names = bus_names

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(pyg_data, str(save_path) + ".pt")

        with open(str(save_path) + ".gpickle", "wb") as f:
            pickle.dump(G, f)

    return G, pyg_data