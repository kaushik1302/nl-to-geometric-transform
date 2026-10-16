import json

import numpy as np
import pandas as pd
import torch

from dyooptix.data import vocab
from dyooptix.data.distributions import RANGES
from dyooptix.models.metrics import evaluate
from dyooptix.models.mlp import REPS, GatedMLP, load_features, predict, standardise

SPLITS = ["test", "test_unseen_templates", "geobench"]
SEEN_VERBS = vocab.TRANSLATE_VERBS + vocab.ROTATE_VERBS + vocab.SCALE_UP_VERBS + vocab.SCALE_DOWN_VERBS


def load_rows(split):
    with open(f"data/splits/{split}.jsonl") as f:
        return [json.loads(line) for line in f]


def oracle(row):
    """Perfect reader: true direction and level, midpoint of the range for vague amounts."""
    t = row["target"]
    pred = [0.0, 0.0, 1.0, 0.0, 0.0]
    for kind, level in zip(row["edits"], row["levels"]):
        if level == "explicit":
            size = None
        else:
            low, high = RANGES[kind][level]
            size = (low + high) / 2

        if kind == "translate":
            if size is None:
                pred[0], pred[1] = t[0], t[1]
            else:
                length = np.hypot(t[0], t[1])
                pred[0], pred[1] = t[0] / length * size, t[1] / length * size
        elif kind == "rotate":
            if size is None:
                pred[2], pred[3] = t[2], t[3]
            else:
                angle = np.radians(np.sign(t[3]) * size)
                pred[2], pred[3] = np.cos(angle), np.sin(angle)
        else:
            pred[4] = t[4] if size is None else np.sign(t[4]) * size
    return pred


def group_labels(row, split):
    labels = {"all": "all", "category": row["category"]}
    if row["category"] != "combined":
        labels["level"] = row["levels"][0]
    if split == "geobench":
        labels["adverb_seen"] = str(row["adverb_seen"])
        labels["verb_seen"] = str(row["instruction"].split()[0] in SEEN_VERBS)
    return labels


def load_model(rep):
    checkpoint = torch.load(f"outputs/mlp_{rep}.pt")
    model = GatedMLP(len(checkpoint["mean"]))
    model.load_state_dict(checkpoint["state"])
    return model, checkpoint["mean"].numpy(), checkpoint["std"].numpy()


def main():
    results = []
    for split in SPLITS:
        rows = load_rows(split)
        y = np.array([r["target"] for r in rows])
        presence = np.array([r["presence"] for r in rows])
        labels = [group_labels(r, split) for r in rows]

        predictions = {}  # model name -> (values, presence flags)
        for rep in REPS:
            model, mean, std = load_model(rep)
            X = load_features(split, rep)
            predictions[f"mlp_{rep}"] = predict(model, standardise(X, mean, std))
        if split != "geobench":
            predictions["oracle"] = (np.array([oracle(r) for r in rows]), presence)

        pairs = sorted({(g, v) for lab in labels for g, v in lab.items()})
        for group, value in pairs:
            mask = np.array([lab.get(group) == value for lab in labels])
            for name, (pred, present) in predictions.items():
                scores = evaluate(y[mask], pred[mask])
                scores["presence_acc"] = (present[mask] == presence[mask]).all(axis=1).mean()
                results.append({"model": name, "split": split, "group": group, "value": value,
                                "n": int(mask.sum()), **scores})

    df = pd.DataFrame(results)
    df.to_csv("outputs/breakdown.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    print("Corner error (px) per group")
    print(df.pivot_table(index=["split", "group", "value", "n"], columns="model", values="corner_px").round(1))
    print("\nGeoBench presence accuracy per group")
    geo = df[df.split == "geobench"]
    print(geo.pivot_table(index=["group", "value", "n"], columns="model", values="presence_acc").round(3))


if __name__ == "__main__":
    main()