import torch

from src.models.st_gnn import STGAT_GRU, STGCN_GRU
from src.models.baselines import MLPBaseline, GCNBaseline, GATBaseline

in_dim = 9

models = {
    "MLP": MLPBaseline(in_dim),
    "GCN": GCNBaseline(in_dim),
    "GAT": GATBaseline(in_dim),
    "STGCN": STGCN_GRU(in_dim),
    "ResiliGraph-STGAT": STGAT_GRU(in_dim),
}

print("=" * 70)
print("MODEL PARAMETER COUNTS")
print("=" * 70)

for name, model in models.items():
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"{name:20s} total={params:,}  trainable={trainable:,}  ({params/1000:.1f}K)")