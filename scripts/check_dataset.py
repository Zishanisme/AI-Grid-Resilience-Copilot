import torch
from pathlib import Path

path = Path("data/processed/phase2/train_graphs.pt")

if not path.exists():
    raise FileNotFoundError("Run Phase 2 first: python scripts\\run_phase2_dataset.py")

graphs = torch.load(path, weights_only=False)

print("Graphs:", len(graphs))
print("Nodes:", graphs[0].num_nodes)
print("Features:", graphs[0].x.shape[1])
print("Edges:", graphs[0].edge_index.shape[1])
print("Labels shape:", graphs[0].y.shape)

y = torch.cat([g.y for g in graphs], dim=0)

print("\nLabel distribution:")
print("fault:", y[:, 0].mean().item())
print("congestion:", y[:, 1].mean().item())
print("voltage:", y[:, 2].mean().item())

if torch.isnan(graphs[0].x).any():
    print("WARNING: NaN in features")

if y[:, 0].mean().item() in [0.0, 1.0]:
    print("WARNING: fault labels may be degenerate")

if y[:, 1].mean().item() in [0.0, 1.0]:
    print("WARNING: congestion labels may be degenerate")

if y[:, 2].mean().item() in [0.0, 1.0]:
    print("WARNING: voltage labels may be degenerate")