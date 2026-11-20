"""Object-level instruction -> one edit for the object that moves; translations come from the scene when a relation asks.
Rules: the first-named object moves and "it" refers to it; rotation and scale are applied (and clamped) before a
relation is measured; a relation decides the final position, replacing any plain move."""
import re

import cv2
import numpy as np

from dyooptix.geom.affine import build_affine
from dyooptix.geom.clamp import clamp
from dyooptix.geom.contact import relation_shift
from dyooptix.geom.grounding import outline
from dyooptix.models.relations import name, parse
from dyooptix.models.split import merge, split_clauses

REFERENCE_WIDTH = 512                      # the model's "N px" refers to a 512-wide image
PIXELS = re.compile(r"\d\s*(?:pixels|px)\b")
NOTHING = np.array([0.0, 0.0, 1.0, 0.0, 0.0])


def read(sentence, split=True):
    """(clauses, relation per clause or None, the object that moves, the texts the text model should read)."""
    clauses = (split_clauses(sentence) if split else None) or [sentence]
    rels = [parse(c) for c in clauses]
    names = [r["mover"] if r else name(c) for c, r in zip(clauses, rels)]
    mover = next((n for n in names if n), None)                    # later "it" refers to this object
    texts = [r["rest"] if r else c for c, r in zip(clauses, rels)]  # relation clauses: only the part before it
    return clauses, rels, mover, texts


def scene_edit(clauses, rels, preds, mover_mask, target_mask):
    """preds: text-model rows per clause [tx, ty, cos, sin, log s] (tx, ty fractions of a 512-wide image).
    Returns the edit [tx, ty, cos, sin, log s] with tx, ty as fractions of this photo's width."""
    H, W = mover_mask.shape
    plain = []
    for c, r, p in zip(clauses, rels, preds):
        if r is None:
            p = np.array(p, float)
            if PIXELS.search(c):                                   # "50 px" means 50 pixels of this photo
                p[:2] *= REFERENCE_WIDTH / W
            plain.append(p)
    edit = merge(plain) if plain else NOTHING.copy()
    i = next((k for k, r in enumerate(rels) if r), None)
    if i is None:
        return edit

    ys, xs = np.nonzero(mover_mask > 127)
    centre = (xs.mean(), ys.mean())
    deg = np.degrees(np.arctan2(edit[3], edit[2]))
    _, _, deg, s, _ = clamp(outline(mover_mask), centre, (W, H), 0, 0, deg, np.exp(edit[4]))   # turn and size first
    a = np.radians(deg)
    M = build_affine({"tx": 0, "ty": 0, "cos": np.cos(a), "sin": np.sin(a), "log_s": np.log(s)}, centre, W)
    shaped = cv2.warpAffine(mover_mask, M, (W, H), flags=cv2.INTER_NEAREST)   # the object as it will look
    tx, ty = relation_shift(rels[i], shaped, target_mask, preds[i])
    return np.array([tx / W, ty / W, np.cos(a), np.sin(a), np.log(s)])