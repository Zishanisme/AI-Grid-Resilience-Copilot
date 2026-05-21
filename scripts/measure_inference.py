import time
import torch
from pathlib import Path

from src.models.st_gnn import STGAT_GRU

device = torch.device("cpu")

data_dir = Path("data/processed/phase2")

test_graphs = torch.load(
    data_dir / "test_graphs.pt",
    weights_only=False
)

model = STGAT_GRU(in_dim=9)
model.eval()
model.to(device)

SEQ_LEN = 20

# build one temporal sequence manually from first 20 graphs
seq = test_graphs[:SEQ_LEN]

seq_dev = [g.to(device) for g in seq]

# warmup
with torch.no_grad():
    _ = model(seq_dev)

runs = 50
times = []

for _ in range(runs):
    t0 = time.perf_counter()

    with torch.no_grad():
        _ = model(seq_dev)

    ms = (time.perf_counter() - t0) * 1000
    times.append(ms)

avg_ms = sum(times) / len(times)
min_ms = min(times)
max_ms = max(times)

print("=" * 60)
print("ResiliGraph-STGAT inference timing")
print("=" * 60)
print(f"Sequence length: {SEQ_LEN}")
print(f"Runs: {runs}")
print(f"Average inference time: {avg_ms:.2f} ms")
print(f"Min inference time:     {min_ms:.2f} ms")
print(f"Max inference time:     {max_ms:.2f} ms")
print("=" * 60)