"""Score the demo models once on the human-written test set (data/human/human_instructions.csv)."""
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from dyooptix.data.distributions import RANGES
from dyooptix.models.breakdown import load_model
from dyooptix.models.encode import hand_features
from dyooptix.models.mlp import predict, standardise

REPS = ["hand", "minilm", "hand+minilm"]
KINDS = ["translate", "rotate", "scale"]
D = 0.7071
TRANSLATE_DIRS = {"right": (1, 0), "left": (-1, 0), "up": (0, -1), "down": (0, 1),
                  "up-right": (D, -D), "up-left": (-D, -D), "down-right": (D, D), "down-left": (-D, D)}
DIRS = {"translate": set(TRANSLATE_DIRS), "rotate": {"cw", "ccw"}, "scale": {"up", "down"}}
LEVELS = {"small", "medium", "large"}
MINILM = SentenceTransformer("all-MiniLM-L6-v2")


def wrap(angle):
    return (angle + 180) % 360 - 180


def check(df):
    """Stop with a clear message if any label is not in the agreed format."""
    problems = []
    for i, r in df.iterrows():
        for kind in KINDS:
            d, a = r[f"{kind}_dir"], r[f"{kind}_amount"]
            if d == "" and a == "":
                continue
            if d not in DIRS[kind]:
                problems.append(f"line {i + 2}: {kind}_dir = {d!r}")
            if a not in LEVELS | {"none"}:
                try:
                    float(a)
                except ValueError:
                    problems.append(f"line {i + 2}: {kind}_amount = {a!r}")
    if problems:
        raise SystemExit("Fix these labels first:\n" + "\n".join(problems))


def encode(sentences, rep):
    """Same features as in training; "hand+minilm" stacks hand then MiniLM."""
    parts = []
    for part in rep.split("+"):
        if part == "hand":
            parts.append(np.array([hand_features(s) for s in sentences], dtype=np.float32))
        else:
            parts.append(MINILM.encode(sentences))
    return np.hstack(parts)


def score_edit(kind, direction, amount, pred, detected):
    """Scores for one labelled edit of one sentence."""
    tx, ty, c, s, log_s = pred
    angle = np.degrees(np.arctan2(s, c))
    numeric = amount not in LEVELS | {"none"}
    out = {"detected": detected, "amount": amount, "pred_tx_px": tx * 512, "pred_ty_px": ty * 512,
           "pred_angle": angle, "pred_scale": np.exp(log_s)}

    if kind == "translate":
        dx, dy = TRANSLATE_DIRS[direction]
        length = np.hypot(tx, ty)
        out["direction_ok"] = bool(detected and length > 0
                                   and (tx * dx + ty * dy) / length > np.cos(np.radians(22.5)))
        size = length
        if numeric:
            out["error"] = abs(length * 512 - float(amount))            # pixels
    elif kind == "rotate":
        sign = 1 if direction == "ccw" else -1
        true_angle = wrap(sign * float(amount)) if numeric else sign
        half_turn = numeric and abs(true_angle) > 179                   # 180 deg: either way is right
        out["direction_ok"] = bool(detected and (half_turn or np.sign(angle) == np.sign(true_angle)))
        size = abs(angle)
        if numeric:
            out["error"] = abs(wrap(angle - true_angle))               # degrees
    else:
        sign = 1 if direction == "up" else -1
        out["direction_ok"] = bool(detected and np.sign(log_s) == sign)
        size = abs(log_s)
        if numeric:
            out["error"] = abs(log_s - np.log(float(amount)))          # |log s|

    if amount in LEVELS:
        low, high = RANGES[kind][amount]
        out["level_hit"] = float(low <= size <= high)
    out["amount_type"] = "none" if amount == "none" else "vague" if amount in LEVELS else "explicit"
    return out


def main():
    df = pd.read_csv("data/human/human_instructions.csv", keep_default_na=False, dtype=str)
    check(df)
    sentences = [s.lower().strip().rstrip(".!?") for s in df["instruction"]]
    truth = np.array([[r[f"{k}_dir"] != "" for k in KINDS] for _, r in df.iterrows()])

    edits, presence_rows = [], []
    for rep in REPS:
        model, mean, std = load_model(rep)
        preds, present = predict(model, standardise(encode(sentences, rep), mean, std))
        for i, r in df.iterrows():
            presence_rows.append({"model": rep, "presence_ok": bool((present[i] == truth[i]).all())}) # type: ignore
            for k, kind in enumerate(KINDS):
                if r[f"{kind}_dir"]:
                    edits.append({"model": rep, "line": i + 2, "instruction": r["instruction"], "kind": kind, # type: ignore
                                  **score_edit(kind, r[f"{kind}_dir"], r[f"{kind}_amount"],
                                               preds[i], bool(present[i][k]))}) # type: ignore

    edits = pd.DataFrame(edits)
    edits.to_csv("outputs/human_edits.csv", index=False)
    summary = edits.groupby(["model", "kind"], sort=False).agg(
        n=("detected", "size"), detected=("detected", "mean"), direction_ok=("direction_ok", "mean"),
        explicit_n=("error", "count"), explicit_median_error=("error", "median"),
        vague_n=("level_hit", "count"), level_hit=("level_hit", "mean"))
    summary.to_csv("outputs/human_summary.csv")
    presence = pd.DataFrame(presence_rows).groupby("model", sort=False)["presence_ok"].mean()

    pd.set_option("display.width", 200)
    print(f"{len(df)} sentences, {len(edits) // len(REPS)} labelled edits")
    print("\nSentences with all three presence flags right:")
    print(presence.round(3).to_string())
    print("\nPer edit type (explicit error: translate px, rotate deg, scale |log s|):")
    print(summary.round(3))


if __name__ == "__main__":
    main()