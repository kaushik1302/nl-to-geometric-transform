import copy
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn

from dyooptix.models.metrics import evaluate

REPS = ["hand", "tfidf", "clip", "minilm", "hand+minilm"]
EVAL_SPLITS = ["val", "test", "test_unseen_templates", "geobench"]
METRICS = ["tx_px", "ty_px", "theta_deg", "log_s", "corner_px", "presence_f1"]
# Which edit owns each target column: translate -> tx, ty; rotate -> cos, sin; scale -> log s
COLUMN_EDIT = [0, 0, 1, 1, 2]
DO_NOTHING = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])
SEEDS = [0, 1, 2]  # every model is trained 3 times; results reported as mean ± std

HIDDEN = 256
BATCH = 64
LR = 1e-3
MAX_EPOCHS = 200
PATIENCE = 15


def load_features(split, rep):
    """Saved features; "a+b" stacks two representations side by side (e.g. hand+minilm)."""
    return np.hstack([np.load(f"data/features/{split}_{part}.npy") for part in rep.split("+")])


def load_split(split, rep):
    X = load_features(split, rep)
    with open(f"data/splits/{split}.jsonl") as f:
        rows = [json.loads(line) for line in f]
    y = np.array([r["target"] for r in rows], dtype=np.float32)
    presence = np.array([r["presence"] for r in rows], dtype=np.float32)
    return X, y, presence


def standardise(X, mean, std):
    return torch.tensor((X - mean) / std, dtype=torch.float32)


class GatedMLP(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(n_in, HIDDEN), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
        )
        self.presence_head = nn.Linear(HIDDEN, 3)  # logits: translate, rotate, scale present?
        self.value_head = nn.Linear(HIDDEN, 5)     # tx, ty, cos, sin, log s

    def forward(self, x):
        h = self.trunk(x)
        return self.presence_head(h), self.value_head(h)


def loss_fn(logits, values, y, presence):
    presence_loss = nn.functional.binary_cross_entropy_with_logits(logits, presence)
    mask = presence[:, COLUMN_EDIT]  # 1 where the edit owning that column is present
    value_loss = (torch.abs(values - y) * mask).sum() / mask.sum().clamp(min=1)
    return presence_loss + value_loss


def predict(model, X):
    model.eval()
    with torch.no_grad():
        logits, values = model(X)
    present = logits > 0                             # logit > 0  <=>  probability > 0.5
    pred = torch.where(present[:, COLUMN_EDIT], values, DO_NOTHING)
    return pred.numpy(), present.numpy().astype(int)


def train(rep, seed):
    X_train, y_train, p_train = load_split("train", rep)
    X_val, y_val, p_val = load_split("val", rep)
    mean, std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-6

    Xt, yt, pt = standardise(X_train, mean, std), torch.tensor(y_train), torch.tensor(p_train)
    Xv, yv, pv = standardise(X_val, mean, std), torch.tensor(y_val), torch.tensor(p_val)

    torch.manual_seed(seed)  # fixes initial weights, batch order and dropout for this run
    model = GatedMLP(X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    best_loss, best_state, bad_epochs = np.inf, None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        order = torch.randperm(len(Xt))
        for start in range(0, len(Xt), BATCH):
            idx = order[start:start + BATCH]
            logits, values = model(Xt[idx])
            loss = loss_fn(logits, values, yt[idx], pt[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logits, values = model(Xv)
            val_loss = loss_fn(logits, values, yv, pv).item()
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss, best_state, bad_epochs = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs == PATIENCE:
                break

    print(f"{rep} seed {seed}: stopped at epoch {epoch + 1}, best val loss {best_loss:.4f}")
    model.load_state_dict(best_state)  # type: ignore
    return model, mean, std


def main():
    rows = []
    for rep in REPS:
        for seed in SEEDS:
            model, mean, std = train(rep, seed)
            if seed == 0:  # the seed-0 model is used by breakdown, showcase and the demo
                torch.save({"state": model.state_dict(), "mean": torch.tensor(mean), "std": torch.tensor(std)},
                           f"outputs/mlp_{rep}.pt")
            for split in EVAL_SPLITS:
                X, y, presence = load_split(split, rep)
                pred, present = predict(model, standardise(X, mean, std))
                scores = evaluate(y, pred)
                scores["presence_f1"] = f1_score(presence.astype(int), present, average="macro")
                rows.append({"model": f"mlp_{rep}", "split": split, "seed": seed, **scores})

    df = pd.DataFrame(rows)
    df.to_csv("outputs/mlp.csv", index=False)  # every seed, every split
    summary = df.groupby(["model", "split"], sort=False)[METRICS].agg(["mean", "std"])
    summary.to_csv("outputs/mlp_summary.csv")
    pd.set_option("display.width", 250)
    print(summary.round(3))


if __name__ == "__main__":
    main()