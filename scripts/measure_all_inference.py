import time
import torch
from pathlib import Path

from src.models.st_gnn import STGAT_GRU, STGCN_GRU
from src.models.baselines import MLPBaseline, GCNBaseline, GATBaseline

device = torch.device("cpu")
data_dir = Path("data/processed/phase2")

graphs = torch.load(data_dir / "test_graphs.pt", weights_only=False)

SEQ_LEN = 20
seq = graphs[:SEQ_LEN]
seq_dev = [g.to(device) for g in seq]

models = {
    "MLP": MLPBaseline(9),
    "GCN": GCNBaseline(9),
    "GAT": GATBaseline(9),
    "STGCN": STGCN_GRU(9),
    "ResiliGraph-STGAT": STGAT_GRU(9),
}

print("=" * 80)
print("MODEL EFFICIENCY BENCHMARK")
print("=" * 80)

for name, model in models.items():
    model.to(device)
    model.eval()

    params = sum(p.numel() for p in model.parameters())

    # all project models expect graph_seq
    inp = seq_dev

    # warmup
    with torch.no_grad():
        _ = model(inp)

    runs = 50
    times = []

    for _ in range(runs):
        t0 = time.perf_counter()

        with torch.no_grad():
            _ = model(inp)

        times.append((time.perf_counter() - t0) * 1000)

    print(
        f"{name:20s} "
        f"Params={params/1000:7.1f}K | "
        f"Avg inference={sum(times)/len(times):7.2f} ms | "
        f"Min={min(times):7.2f} ms | "
        f"Max={max(times):7.2f} ms"
    )