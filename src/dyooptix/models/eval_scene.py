"""Scene-derived edits on the shape test set (data/shapes): predicted vs exact expected translation.
Run with the true masks (text model + geometry) and with Grounded-SAM masks (the full system)."""
import sys
import time

import cv2
import numpy as np
import pandas as pd

from dyooptix.geom.clamp import clamp
from dyooptix.geom.grounding import ObjectNotFound, find_object, outline
from dyooptix.models.breakdown import load_model
from dyooptix.models.eval_human import encode
from dyooptix.models.mlp import predict, standardise
from dyooptix.models.numbers import fix_numbers
from dyooptix.models.scene import read, scene_edit

REP = "hand+minilm"
TOL = 3.0                                          # px: a move within 3 px of the exact answer counts as right
SHAPES = "data/shapes"


def text_preds(texts, model, mean, std):
    """Text model on each clause, with exact numbers fixed, as in the demo."""
    pred, present = predict(model, standardise(encode(texts, REP), mean, std))
    return [fix_numbers(t, p, q.astype(bool)) for t, p, q in zip(texts, pred, present)]


def mask(row_id, image, word, source, avoid=None):
    """(mask, box). avoid: the mover's box, so the target is looked for among the other objects."""
    if source == "true":
        return cv2.imread(f"{SHAPES}/masks/{row_id}_{word}.png", cv2.IMREAD_GRAYSCALE), None
    found = find_object(image, word, avoid)
    return found["mask"], found["box"]


def run(source, model, mean, std):
    out = []
    for r in pd.read_csv(f"{SHAPES}/shapes.csv").itertuples():
        image = cv2.cvtColor(cv2.imread(f"{SHAPES}/images/{r.id}.png"), cv2.COLOR_BGR2RGB)
        H, W = image.shape[:2]
        clauses, rels, mover, texts = read(r.instruction)
        rel = next(x for x in rels if x)
        start = time.time()
        try:
            mover_mask, mover_box = mask(r.id, image, mover, source)
            target_mask = mask(r.id, image, rel["target"], source, mover_box)[0] if rel["target"] else None
            edit = scene_edit(clauses, rels, text_preds(texts, model, mean, std), mover_mask, target_mask)
            ys, xs = np.nonzero(mover_mask > 127)
            deg = np.degrees(np.arctan2(edit[3], edit[2]))
            tx, ty, *_ = clamp(outline(mover_mask), (xs.mean(), ys.mean()), (W, H),
                               edit[0] * W, edit[1] * W, deg, np.exp(edit[4]))       # what would really be applied
            reason = ""
        except (ValueError, ObjectNotFound) as error:
            tx = ty = np.nan
            reason = str(error)
        error_px = np.hypot(tx - r.expected_tx, ty - r.expected_ty)
        out.append({"masks": source, "id": r.id, "kind": r.kind, "note": r.note, "instruction": r.instruction,
                    "expected_tx": r.expected_tx, "expected_ty": r.expected_ty, "pred_tx": tx, "pred_ty": ty,
                    "error_px": error_px, "ok": bool(error_px <= TOL), "seconds": time.time() - start,
                    "reason": reason})
        print(f"{source} {r.id}  error {error_px:6.1f} px  {reason}")
    return pd.DataFrame(out)


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 70)
    sources = sys.argv[1:] or ["true", "gsam"]
    model, mean, std = load_model(REP)
    df = pd.concat([run(s, model, mean, std) for s in sources])
    df.to_csv(f"outputs/scene_shapes_{'_'.join(sources)}.csv", index=False)

    print("\nPer sentence type:")
    print(df.groupby(["masks", "kind"]).agg(n=("ok", "size"), within_3px=("ok", "mean"),
                                            median_error_px=("error_px", "median"),
                                            refused=("reason", lambda x: (x != "").sum())).round(3))
    print("\nOverall, within 3 px:")
    print(df.groupby("masks")["ok"].mean().round(3))
    wrong = df[~df.ok]
    if len(wrong):
        print("\nNot within 3 px:")
        print(wrong[["masks", "id", "instruction", "expected_tx", "expected_ty", "pred_tx", "pred_ty", "reason"]]
              .round(1).to_string(index=False))


if __name__ == "__main__":
    main()