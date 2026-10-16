import csv
import json

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dyooptix.models.metrics import evaluate

REPS = ["hand", "tfidf", "clip", "minilm"]
EVAL_SPLITS = ["val", "test", "test_unseen_templates", "geobench"]
ALPHAS = [0.1, 1, 10, 100, 1000]
COLUMNS = ["model", "split", "tx_px", "ty_px", "theta_deg", "log_s", "corner_px"]


def load_targets(split):
    with open(f"data/splits/{split}.jsonl") as f:
        return np.array([json.loads(line)["target"] for line in f])


def load_features(split, rep):
    return np.load(f"data/features/{split}_{rep}.npy")


def make_ridge(alpha):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def main():
    targets = {split: load_targets(split) for split in ["train"] + EVAL_SPLITS}
    rows = []

    # Constant baseline: always predict the train median, whatever the sentence says
    median = np.median(targets["train"], axis=0)
    for split in EVAL_SPLITS:
        pred = np.tile(median, (len(targets[split]), 1))
        rows.append({"model": "constant", "split": split, **evaluate(targets[split], pred)})

    # Ridge on each representation, alpha chosen on val
    for rep in REPS:
        X_train = load_features("train", rep)
        X_val = load_features("val", rep)
        best_alpha, best_score = None, np.inf
        for alpha in ALPHAS:
            model = make_ridge(alpha).fit(X_train, targets["train"])
            score = evaluate(targets["val"], model.predict(X_val))["corner_px"]
            if score < best_score:
                best_alpha, best_score = alpha, score
        print(f"{rep}: best alpha = {best_alpha}")

        model = make_ridge(best_alpha).fit(X_train, targets["train"])
        for split in EVAL_SPLITS:
            pred = model.predict(load_features(split, rep))
            rows.append({"model": f"ridge_{rep}", "split": split, **evaluate(targets[split], pred)})

    print(f"\n{'model':14}{'split':24}{'tx_px':>8}{'ty_px':>8}{'θ_deg':>8}{'log_s':>8}{'corner':>9}")
    for r in rows:
        print(f"{r['model']:14}{r['split']:24}{r['tx_px']:8.1f}{r['ty_px']:8.1f}"
              f"{r['theta_deg']:8.2f}{r['log_s']:8.3f}{r['corner_px']:9.1f}")

    with open("outputs/baselines.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()