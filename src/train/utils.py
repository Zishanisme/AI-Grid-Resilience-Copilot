import torch

def make_sequences(graphs, seq_len=6):
    sequences = []
    for i in range(seq_len - 1, len(graphs)):
        sequences.append(graphs[i - seq_len + 1:i + 1])
    return sequences


def save_checkpoint(model, path):
    import torch
    torch.save(model.state_dict(), path)

def load_checkpoint(model, path, device="cpu"):
    import torch
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    return model

def check_label_balance(train_graphs, val_graphs=None, test_graphs=None):
    import torch

    def compute_stats(graphs, name):
        y = torch.cat([g.y for g in graphs], dim=0)

        print(f"\n{name} label balance:")
        print(f"  fault_risk:        {y[:, 0].mean().item():.4f}")
        print(f"  congestion_risk:   {y[:, 1].mean().item():.4f}")
        print(f"  voltage_violation: {y[:, 2].mean().item():.4f}")

    compute_stats(train_graphs, "Train")

    if val_graphs is not None:
        compute_stats(val_graphs, "Validation")

    if test_graphs is not None:
        compute_stats(test_graphs, "Test")