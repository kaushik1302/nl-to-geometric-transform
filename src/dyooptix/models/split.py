"""Clause splitting: predict each clause of a combined instruction separately, then merge.

Inference only: uses the trained models as they are. Our parameters mean "rotate and scale
about the object's own centre, then move by t", so applying edits one after another gives
t = t1 + t2, angle = angle1 + angle2 and log s = log s1 + log s2, in any order.
"""
import re

import numpy as np

from dyooptix.models.mlp import predict, standardise

# Joining words, longest first so ", and then" is not cut as "," + "and then"
JOINER = re.compile(r"\s*(?:,\s*and then\b|,\s*then\b|,\s*and\b|\band then\b|\bthen\b|\band\b|[,;])\s*")


# "up and right", "up and to the left": one diagonal direction, not two edits
DIR = r"(?:to the )?(?:up|down|left|right)(?:wards?)?"
DIAGONAL = re.compile(rf"\b({DIR}) and ({DIR})\b")
KEEP = "\x00"   # stands in for a protected " and " while splitting


def split_clauses(sentence):
    """'move it left and then rotate it' -> ['move it left', 'rotate it']"""
    protected = DIAGONAL.sub(rf"\1{KEEP}\2", sentence)
    return [c.replace(KEEP, " and ") for c in JOINER.split(protected) if c.strip()]


def merge(preds):
    """Gated predictions, one row per clause [tx, ty, cos, sin, log s] -> one row."""
    preds = np.asarray(preds, dtype=float)
    angle = np.arctan2(preds[:, 3], preds[:, 2]).sum()    # do-nothing rows add 0
    return np.array([preds[:, 0].sum(), preds[:, 1].sum(),
                     np.cos(angle), np.sin(angle), preds[:, 4].sum()])


def predict_split(model, mean, std, sentence, featurise):
    """featurise(list of texts) -> feature rows, built exactly as in training."""
    clauses = split_clauses(sentence) or [sentence]
    preds, present = predict(model, standardise(featurise(clauses), mean, std))
    return merge(preds), present.max(axis=0), clauses   # an edit is present if any clause has it