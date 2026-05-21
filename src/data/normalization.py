import torch
import copy

def fit_feature_stats(graphs):
    x_all = torch.cat([g.x for g in graphs], dim=0)
    edge_all = torch.cat([g.edge_attr for g in graphs], dim=0)

    return {
        "x_mean": x_all.mean(dim=0),
        "x_std": x_all.std(dim=0).clamp(min=1e-6),
        "edge_mean": edge_all.mean(dim=0),
        "edge_std": edge_all.std(dim=0).clamp(min=1e-6),
    }


def apply_feature_stats(graphs, stats):
    normalized = []

    for g in graphs:
        g = copy.deepcopy(g)
        g.x = (g.x - stats["x_mean"]) / stats["x_std"]
        g.edge_attr = (g.edge_attr - stats["edge_mean"]) / stats["edge_std"]

        assert not torch.isnan(g.x).any(), "NaN found in node features"
        assert not torch.isnan(g.edge_attr).any(), "NaN found in edge features"

        normalized.append(g)

    return normalized