import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset

from dyooptix.geom.warp import iou, warp
from dyooptix.models.breakdown import SEEN_VERBS, load_model
from dyooptix.models.mlp import REPS, load_features, predict, standardise

SIZE = 512
SHOW_MODELS = ["hand", "minilm", "hand+minilm"]
COLUMNS = REPS + ["do_nothing"]
RED, GREEN = (255, 0, 0), (0, 255, 0)


def load_mask(pil_mask):
    mask = np.array(pil_mask.convert("L"))
    mask = cv2.resize(mask, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)  # ori_mask is 640x640
    return (mask > 127).astype(np.uint8)


def outline(image, mask, colour):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.drawContours(image.copy(), contours, -1, colour, 3)


def main():
    with open("data/splits/geobench.jsonl") as f:
        rows = [json.loads(line) for line in f]
    y = np.array([r["target"] for r in rows])

    preds = {}
    for rep in REPS:
        model, mean, std = load_model(rep)
        X = load_features("geobench", rep)
        preds[rep], _ = predict(model, standardise(X, mean, std))

    data = load_dataset("CIawevy/GeoBench", "2d", cache_dir="data/hf_cache")["data"]

    # 1. Transform-consistency IoU on every GeoBench row
    masks = data.select_columns(["ori_mask"])  # type: ignore # decode only the mask column: much faster
    results = []
    for i, r in enumerate(rows):
        mask = load_mask(masks[r["geobench_index"]]["ori_mask"])
        true_mask = warp(mask, y[i], cv2.INTER_NEAREST)
        result = {"category": r["category"], "verb_seen": r["instruction"].split()[0] in SEEN_VERBS,
                  "do_nothing": iou(true_mask, mask)}
        for rep in REPS:
            result[rep] = iou(true_mask, warp(mask, preds[rep][i], cv2.INTER_NEAREST))
        results.append(result)

    df = pd.DataFrame(results)
    df.to_csv("outputs/iou_geobench.csv", index=False)
    print("Mean IoU, all rows:", df[COLUMNS].mean().round(3).to_dict())
    print(df.groupby("category")[COLUMNS].mean().round(3))
    print(df.groupby("verb_seen")[COLUMNS].mean().round(3))

    # 2. Qualitative grid: 2 random examples per edit type
    rng = np.random.default_rng(0)
    picks = []
    for category in ["translate", "rotate", "scale"]:
        candidates = [j for j, r in enumerate(rows) if r["category"] == category]
        picks += [int(j) for j in rng.choice(candidates, 2, replace=False)]

    n_cols = 2 + len(SHOW_MODELS)
    fig, axes = plt.subplots(len(picks), n_cols, figsize=(4 * n_cols, 4.4 * len(picks)))
    for row_axes, i in zip(axes, picks):
        sample = data[rows[i]["geobench_index"]]
        image = np.array(sample["ori_img"].convert("RGB"))
        mask = load_mask(sample["ori_mask"])
        true_mask = warp(mask, y[i], cv2.INTER_NEAREST)

        panels = [(f"original\n\"{rows[i]['instruction']}\"", outline(image, mask, GREEN)),
                  ("true edit", outline(warp(image, y[i], cv2.INTER_LINEAR), true_mask, RED))]
        for rep in SHOW_MODELS:
            pred_mask = warp(mask, preds[rep][i], cv2.INTER_NEAREST)
            panel = outline(warp(image, preds[rep][i], cv2.INTER_LINEAR), true_mask, RED)
            panel = outline(panel, pred_mask, GREEN)
            panels.append((f"MLP {rep}   IoU {iou(true_mask, pred_mask):.2f}", panel))

        for ax, (title, panel) in zip(row_axes, panels):
            ax.imshow(panel)
            ax.set_title(title, fontsize=9)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig("outputs/qualitative_geobench.png", dpi=80)
    print("Saved outputs/qualitative_geobench.png")


if __name__ == "__main__":
    main()