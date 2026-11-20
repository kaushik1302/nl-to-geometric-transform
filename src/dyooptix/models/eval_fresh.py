"""Score the system ONCE on the fresh LLM-written test set (data/human_fresh).
Part A: whole-image edits on a 512-px image. Part B: object edits on real photos, run through the demo's pipeline.
Nothing is tuned on these results. Usage: python -m dyooptix.models.eval_fresh [folder] [a|b|ab]"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from dyooptix.models.eval_human import KINDS, check, score_edit
from dyooptix.models.pipeline import _whole, normalise, run
from dyooptix.models.scene import read

COLUMNS = {"move": "translate", "rotate": "rotate", "resize": "scale"}   # fresh column prefix -> the human scorer's
WORDS = {"clockwise": "cw", "anticlockwise": "ccw", "bigger": "up", "smaller": "down"}
VAGUE = ("small", "medium", "large")
REFERENCE = 512                                                         # explicit pixels refer to a 512-px-wide image


def factor(amount, direction, sentence):
    """'75%' -> '0.75' for 'to 75%'. For 'by n%' the number is a change: bigger 1 + n/100, smaller 1 - n/100."""
    if not amount.endswith("%"):
        return amount
    n = float(amount[:-1]) / 100
    if re.search(r"\bby\b", sentence.lower()):
        n = 1 + n if direction == "bigger" else 1 - n
    return f"{n:.4g}"


def load(folder, name):
    """The fresh CSV with the human scorer's column names and words, checked for unknown labels."""
    df = pd.read_csv(f"{folder}/{name}.csv", keep_default_na=False, dtype=str)
    for old, new in COLUMNS.items():
        df = df.rename(columns={f"{old}_direction": f"{new}_dir", f"{old}_amount": f"{new}_amount"})
    df["scale_amount"] = [factor(a, d, s) for a, d, s in zip(df["scale_amount"], df["scale_dir"], df["instruction"])]
    df["rotate_dir"] = df["rotate_dir"].replace(WORDS)
    df["scale_dir"] = df["scale_dir"].replace(WORDS)
    check(df)
    return df


def row_of(p, width):
    """The final edit as the [tx, ty, cos, sin, log s] row score_edit reads, with tx, ty as fractions of `width`."""
    a = np.radians(p["deg"])
    return [p["tx"] / width, p["ty"] / width, np.cos(a), np.sin(a), np.log(p["s"])]


def detected(p):
    """Which edits are really in the final output: translate, rotate, scale."""
    return [np.hypot(p["tx"], p["ty"]) > 0.5, abs(p["deg"]) > 0.05, abs(np.log(p["s"])) > 0.002]


def part_a(r, folder):
    sentence = normalise(r["instruction"].lower().strip().rstrip(".!?"))
    out = _whole(np.zeros((REFERENCE, REFERENCE, 3), np.uint8), sentence, read(sentence)[0], "hand+minilm", "")
    return out["params"], REFERENCE, {}


def part_b(r, folder):
    image = np.array(Image.open(f"{folder}/photos/{r['photo']}.jpg").convert("RGB"))
    try:
        out = run(image, r["instruction"])
    except Exception as error:                                   # one bad row must not lose a long run
        out = {"report": f"{type(error).__name__}: {error}"}
    p = out.get("params")
    extra = {"photo": r["photo"], "object": r["object"], "found_object": out.get("object"),
             "object_ok": out.get("object") == r["object"], "clamped": bool(p and p["warnings"]),
             "reason": "" if p else out["report"]}
    return p, image.shape[1], extra


def score(df, evaluate, folder):
    edits, flags = [], []
    for i, r in df.iterrows():
        p, width, extra = evaluate(r, folder)
        labelled = [bool(r[f"{k}_dir"]) for k in KINDS]
        found = [False] * 3 if p is None else detected(p)
        flags.append({"line": i + 2, "presence_ok": found == labelled, "failed": p is None, **extra})
        for k, kind in enumerate(KINDS):
            if not labelled[k]:
                continue
            amount = r[f"{kind}_amount"]
            row = row_of(p, width if amount in VAGUE else REFERENCE) if p else [0, 0, 1, 0, 0]
            edits.append({"line": i + 2, "instruction": r["instruction"], "kind": kind, "failed": p is None,
                          **score_edit(kind, r[f"{kind}_dir"], amount, row, bool(found[k])), **extra})
        print(i + 2, r["instruction"])
    return pd.DataFrame(edits), pd.DataFrame(flags)


def run_part(tag, df, evaluate, folder, by):
    edits, flags = score(df, evaluate, folder)
    edits.loc[edits["failed"], "error"] = np.nan                 # a failed row has no size to compare
    edits.to_csv(f"outputs/fresh_{tag}_edits.csv", index=False)
    flags.to_csv(f"outputs/fresh_{tag}_sentences.csv", index=False)
    summary = edits.groupby(by, sort=False).agg(
        n=("detected", "size"), detected=("detected", "mean"), direction_ok=("direction_ok", "mean"),
        explicit_n=("error", "count"), explicit_median_error=("error", "median"),
        vague_n=("level_hit", "count"), level_hit=("level_hit", "mean"))
    summary.to_csv(f"outputs/fresh_{tag}_summary.csv")
    print(f"\n=== {tag}: {len(df)} sentences, {len(edits)} labelled edits ===")
    print(f"All presence flags right: {flags['presence_ok'].mean():.3f}   failed rows: {int(flags['failed'].sum())}")
    if "object_ok" in flags:
        print(f"Right object found: {flags['object_ok'].mean():.3f}   rows with a clamp warning: {int(flags['clamped'].sum())}")
    print("Per edit type (explicit error: translate px, rotate deg, scale |log s|):")
    print(summary.round(3))


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "data/human_fresh"
    parts = sys.argv[2] if len(sys.argv) > 2 else "ab"
    tag = Path(folder).name
    pd.set_option("display.width", 200)
    if "a" in parts:
        run_part(f"{tag}_a", load(folder, "part_a_whole_image_edits"), part_a, folder, ["kind"])
    if "b" in parts:
        run_part(f"{tag}_b", load(folder, "part_b_object_edits"), part_b, folder, ["clamped", "kind"])


if __name__ == "__main__":
    main()