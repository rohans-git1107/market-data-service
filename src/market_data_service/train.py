"""Train baselines and a small neural network to predict price direction.

Run with:
    python -m market_data_service.train data/snapshot.csv

Steps: label the data -> split by time -> baselines -> small neural network.
"""

import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from market_data_service.label_data import label_data

FEATURES = ["price_vs_vwap", "imbalance", "volume_rate"]
TRAIN_FRAC = 0.6
VAL_FRAC = 0.2           # the rest (0.2) is the test set
GAP_MS = 300_000         # 5 minute buffer between splits (windows overlap in time)
SEED = 0


def make_features(df):
    """Turn raw columns into model inputs. volume_rate is very skewed, so log it."""
    X = df[FEATURES].copy()
    X["volume_rate"] = np.log1p(X["volume_rate"])
    return X.to_numpy(dtype=np.float64)


def time_split(df, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC, gap_ms=GAP_MS):
    """Split by time: earliest rows train, next validate, latest test.

    A gap is left between the parts. Neighbouring rows share most of their
    1,000-trade window, so without a gap the test period would leak into
    training.
    """
    df = df.sort_values("time_ms").reset_index(drop=True)
    t = df["time_ms"].to_numpy()
    t_min, t_max = t[0], t[-1]
    span = t_max - t_min

    train_end = t_min + train_frac * span
    val_start = train_end + gap_ms
    val_end = t_min + (train_frac + val_frac) * span
    test_start = val_end + gap_ms

    train = df[t <= train_end]
    val = df[(t >= val_start) & (t <= val_end)]
    test = df[t >= test_start]
    return train, val, test


def prepare(df):
    """Label, split, and standardise using training statistics only."""
    labelled = label_data(df)
    train, val, test = time_split(labelled)

    X_train, X_val, X_test = make_features(train), make_features(val), make_features(test)
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    std[std == 0] = 1.0  # avoid dividing by zero for a constant column

    def scale(X):
        return (X - mean) / std

    return (
        (scale(X_train), train["label"].to_numpy()),
        (scale(X_val), val["label"].to_numpy()),
        (scale(X_test), test["label"].to_numpy()),
    )


def accuracy(pred, y):
    return float((pred == y).mean())


def run_baselines(train, val, test):
    """Return {name: (val_accuracy, test_accuracy)} for the simple baselines."""
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = train, val, test
    results = {}

    # 1. Always predict whichever label was more common in training.
    majority = int(y_tr.mean() >= 0.5)
    results["majority class"] = (
        accuracy(np.full_like(y_va, majority), y_va),
        accuracy(np.full_like(y_te, majority), y_te),
    )

    # 2. Logistic regression: the simplest model that uses the features.
    logreg = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
    results["logistic regression"] = (
        accuracy(logreg.predict(X_va), y_va),
        accuracy(logreg.predict(X_te), y_te),
    )
    return results


def train_mlp(train, val, test, epochs=30, batch_size=256, lr=1e-3):
    """Small neural network. Returns (val_accuracy, test_accuracy)."""
    import torch
    from torch import nn

    torch.manual_seed(SEED)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = train, val, test

    def to_tensors(X, y):
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    X_tr, y_tr = to_tensors(X_tr, y_tr)
    X_va, y_va = to_tensors(X_va, y_va)
    X_te, y_te = to_tensors(X_te, y_te)

    model = nn.Sequential(
        nn.Linear(X_tr.shape[1], 32), nn.ReLU(),
        nn.Linear(32, 32), nn.ReLU(),
        nn.Linear(32, 1),
    )
    loss_fn = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss, best_state = float("inf"), None
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(X_tr))
        for i in range(0, len(order), batch_size):
            batch = order[i : i + batch_size]
            optimiser.zero_grad()
            loss = loss_fn(model(X_tr[batch]).squeeze(1), y_tr[batch])
            loss.backward()
            optimiser.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_va).squeeze(1), y_va).item()
        if val_loss < best_val_loss:  # keep the epoch that did best on validation
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch + 1:2d}  validation loss {val_loss:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_va = (model(X_va).squeeze(1) > 0).float()
        pred_te = (model(X_te).squeeze(1) > 0).float()
    return float((pred_va == y_va).float().mean()), float((pred_te == y_te).float().mean())


def main(path):
    df = pd.read_csv(path)
    train, val, test = prepare(df)
    print(f"Rows  train: {len(train[1])}  validation: {len(val[1])}  test: {len(test[1])}")
    print(f"Fraction 'up'  train: {train[1].mean():.3f}  validation: {val[1].mean():.3f}  "
          f"test: {test[1].mean():.3f}\n")

    results = run_baselines(train, val, test)
    results["neural network"] = train_mlp(train, val, test)

    print(f"\n{'model':<22}{'validation':>12}{'test':>10}")
    for name, (va, te) in results.items():
        print(f"{name:<22}{va:>12.3f}{te:>10.3f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python -m market_data_service.train data/snapshot.csv")
    main(sys.argv[1])