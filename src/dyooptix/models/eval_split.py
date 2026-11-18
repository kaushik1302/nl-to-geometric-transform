"""Whole sentence vs clause splitting, with the same trained models (inference only)."""
import json

import numpy as np
import pandas as pd

from dyooptix.models.breakdown import load_model
from dyooptix.models.eval_human import KINDS, REPS, check, encode, score_edit
from dyooptix.models.metrics import corner_error
from dyooptix.models.mlp import predict, standardise
from dyooptix.models.split import merge, split_clauses

SPLITS = ["test", "test_unseen_templates", "geobench"]
MODES = ["whole", "split"]


def predict_all(model, mean, std, sentences, rep, use_split):
    """Predictions for many sentences. With use_split, all clauses are encoded in one batch."""
    if not use_split:
        return predict(model, standardise(encode(sentences, rep), mean, std))
    clauses = [split_clauses(s) or [s] for s in sentences]
    flat = [c for cs in clauses for c in cs]
    pred, present = predict(model, standardise(encode(flat, rep), mean, std))
    preds, presents, i = [], [], 0
    for cs in clauses:                       # regroup the clause rows sentence by sentence
        n = len(cs)
        preds.append(merge(pred[i:i + n]))
        presents.append(present[i:i + n].max(axis=0))
        i += n
    return np.array(preds), np.array(presents)


def synthetic():
    out = []
    for split in SPLITS:
        with open(f"data/splits/{split}.jsonl", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        sentences = [r["instruction"] for r in rows]
        y = np.array([r["target"] for r in rows])
        combined = np.array([r["category"] == "combined" for r in rows])
        for rep in REPS:
            model, mean, std = load_model(rep)
            for mode in MODES:
                pred, _ = predict_all(model, mean, std, sentences, rep, mode == "split")
                for group, keep in [("all", np.ones(len(rows), bool)), ("combined", combined), ("single", ~combined)]:
                    if keep.any():
                        out.append({"split": split, "group": group, "n": int(keep.sum()), "model": rep,
                                    "mode": mode, "corner_px": corner_error(y[keep], pred[keep])})
    return pd.DataFrame(out)


def human():
    df = pd.read_csv("data/human/human_instructions.csv", keep_default_na=False, dtype=str)
    check(df)
    sentences = [s.lower().strip().rstrip(".!?") for s in df["instruction"]]
    truth = np.array([[r[f"{k}_dir"] != "" for k in KINDS] for _, r in df.iterrows()])
    multi = truth.sum(axis=1) > 1            # sentences with two or more labelled edits

    sentence_rows, edit_rows = [], []
    for rep in REPS:
        model, mean, std = load_model(rep)
        for mode in MODES:
            pred, present = predict_all(model, mean, std, sentences, rep, mode == "split")
            for i, r in df.iterrows():
                sentence_rows.append({"model": rep, "mode": mode, "multi": multi[i],
                                      "presence_ok": bool((present[i] == truth[i]).all())}) # type: ignore
                for k, kind in enumerate(KINDS):
                    if r[f"{kind}_dir"]:
                        s = score_edit(kind, r[f"{kind}_dir"], r[f"{kind}_amount"], pred[i], bool(present[i][k])) # type: ignore
                        edit_rows.append({"model": rep, "mode": mode, "multi": multi[i], "kind": kind,
                                          "direction_ok": s["direction_ok"], "error": s.get("error", np.nan)})
    return pd.DataFrame(sentence_rows), pd.DataFrame(edit_rows)


def main():
    pd.set_option("display.width", 200)
    syn = synthetic()
    syn.to_csv("outputs/split_synthetic.csv", index=False)
    print("Corner error (px), whole vs split:")
    print(syn.pivot_table(index=["split", "group", "n"], columns=["model", "mode"], values="corner_px").round(1))

    sentences, edits = human()
    edits.to_csv("outputs/split_human.csv", index=False)
    print("\nHuman set (development only). All three presence flags right:")
    print(sentences.pivot_table(index="multi", columns=["model", "mode"], values="presence_ok").round(3))
    print("\nDirection right, per edit type and multi-edit sentences:")
    print(edits.pivot_table(index=["multi", "kind"], columns=["model", "mode"], values="direction_ok").round(3))
    print("\nExplicit amount, median error (translate px, rotate deg, scale |log s|):")
    print(edits.pivot_table(index="kind", columns=["model", "mode"], values="error", aggfunc="median").round(3))


if __name__ == "__main__":
    main()