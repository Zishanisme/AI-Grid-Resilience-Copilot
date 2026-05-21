def temporal_split(graphs, train_ratio=0.7, val_ratio=0.15):
    n = len(graphs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = graphs[:n_train]
    val = graphs[n_train:n_train + n_val]
    test = graphs[n_train + n_val:]

    return train, val, test