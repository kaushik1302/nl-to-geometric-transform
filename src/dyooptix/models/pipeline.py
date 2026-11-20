"""One entry point for the demo and the scorers: instruction + photo -> edited photo, report and marks to draw."""
import re
from functools import lru_cache

import cv2
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from dyooptix.geom.affine import build_affine
from dyooptix.geom.grounding import ObjectNotFound, find_main, find_object
from dyooptix.geom.object_edit import edit_object
from dyooptix.geom.warp import warp
from dyooptix.models.breakdown import load_model
from dyooptix.models.encode import hand_features
from dyooptix.models.metrics import to_params
from dyooptix.models.mlp import predict, standardise
from dyooptix.models.numbers import fix_numbers
from dyooptix.models.relations import names_whole
from dyooptix.models.scene import read, scene_edit
from dyooptix.models.split import merge

REPS = ["hand+minilm", "hand", "minilm"]   # the first one is the default (used automatically)
EDITS = ["translate", "rotate", "scale"]
REFERENCE_WIDTH = 512                      # explicit pixels in training refer to a 512-wide image
PIXELS = re.compile(r"\d\s*(?:pixels|px)\b")

# Spelling variants of the same word -> the spelling used in training (input cleaning only)
SPELLINGS = {
    "clock-wise": "clockwise", "clock wise": "clockwise",
    "anti-clockwise": "anticlockwise", "anti clockwise": "anticlockwise",
    "counterclockwise": "counter-clockwise", "counter clockwise": "counter-clockwise",
}


@lru_cache(maxsize=None)
def models():
    """Text models and MiniLM, loaded on first use and then kept."""
    return {rep: load_model(rep) for rep in REPS}, SentenceTransformer("all-MiniLM-L6-v2")


def normalise(sentence):
    for variant, canonical in SPELLINGS.items():
        sentence = sentence.replace(variant, canonical)
    return sentence


def encode(text, rep):
    """Same features as in training; "hand+minilm" stacks hand then MiniLM, like load_features."""
    parts = []
    for part in rep.split("+"):
        if part == "hand":
            parts.append(np.array([hand_features(text)], dtype=np.float32))
        else:
            parts.append(models()[1].encode([text]))
    return np.hstack(parts)


def predict_rows(texts, rep):
    """Text model on each text, exact numbers fixed. Returns rows [tx, ty, cos, sin, log s] and per-edit probabilities."""
    model, mean, std = models()[0][rep]
    x = standardise(np.vstack([encode(t, rep) for t in texts]), mean, std)
    pred, present = predict(model, x)
    pred = np.array([fix_numbers(t, p, r.astype(bool)) for t, p, r in zip(texts, pred, present)])
    with torch.no_grad():
        probs = torch.sigmoid(model(x)[0]).max(dim=0).values.numpy()   # per edit: highest over clauses
    return pred, probs


def run(image, instruction, rep=REPS[0], use_split=True, object_name=""):
    """image: RGB uint8 (H, W, 3). Returns {"image", "report", "marks", "mode"}.
    mode: "object" or "whole". marks: boxes to draw on the original photo, only when the main object was a tie."""
    sentence = normalise(instruction.lower().strip().rstrip(".!?"))
    if not sentence:
        return {"image": image, "report": "Type an instruction.", "marks": [], "mode": "whole"}
    clauses, rels, mover, texts = read(sentence, use_split)

    named = object_name.strip().lower() or mover       # the Object box replaces the mover only, not a relation's target
    note, marks, found, label = "", [], None, named
    try:
        if named:
            found = find_object(image, named)
        elif not names_whole(sentence):                # nothing named and no "image / picture": use the main object
            main = find_main(image)
            note, label = main["message"], "main object"
            if main["status"] == "tie":
                marks = [{"box": c["box"], "label": ("edited  " if k == 0 else "") + f"{c['score']:.2f}"}
                         for k, c in enumerate(main["candidates"])]
            found = main if main["mask"] is not None else None
    except ObjectNotFound as error:
        return {"image": image, "report": str(error), "marks": [], "mode": "object"}

    if found is None:
        return _whole(image, sentence, clauses, rep, note)
    return _object(image, sentence, clauses, rels, texts, found, label, rep, note, marks)


def _head(rep, sentence, clauses, probs, note):
    lines = [note] if note else []
    lines += [f"Model: {rep}", f"Text used: {sentence}"]
    if len(clauses) > 1:
        lines.append("Clauses: " + " | ".join(clauses))
    lines.append("Detected edits (probability):")
    lines += [f"  {name:9}  {prob:.2f}  {'✓' if prob > 0.5 else ''}" for name, prob in zip(EDITS, probs)]
    return lines


def _whole(image, sentence, clauses, rep, note):
    pred, probs = predict_rows(clauses, rep)
    target = merge(pred)                               # one clause: unchanged
    h, w = image.shape[:2]
    explicit_px = bool(PIXELS.search(sentence))
    if explicit_px:                                    # tx, ty are fractions of a 512-px width; "50 px" means 50 real pixels
        target[0] *= REFERENCE_WIDTH / w
        target[1] *= REFERENCE_WIDTH / w
    p = to_params(target)
    angle = np.degrees(np.arctan2(p["sin"], p["cos"]))
    M = build_affine(p, (w / 2, h / 2), w)
    lines = _head(rep, sentence, clauses, probs, note)
    lines += ["Edited: the whole image",
              f"Translation: tx = {p['tx'] * w:+.0f} px, ty = {p['ty'] * w:+.0f} px   (+x right, +y down)"
              + ("   [explicit pixels: real pixels of this image]" if explicit_px else ""),
              f"Rotation:    {angle:+.1f}°   (+ = counter-clockwise)",
              f"Scale:       ×{np.exp(p['log_s']):.2f}",
              f"Image size:  {w} × {h}",
              "Affine matrix (2×3):",
              np.array2string(M, precision=2, suppress_small=True)]
    return {"image": warp(image, target, cv2.INTER_LINEAR), "report": "\n".join(lines), "marks": [], "mode": "whole",
            "params": {"tx": p["tx"] * w, "ty": p["ty"] * w, "deg": angle, "s": float(np.exp(p["log_s"])), "warnings": []}}


def _object(image, sentence, clauses, rels, texts, found, label, rep, note, marks):
    rel = next((r for r in rels if r), None)
    try:
        target_mask = find_object(image, rel["target"], found["box"])["mask"] if rel and rel["target"] else None
        pred, probs = predict_rows(texts, rep)
        edit = scene_edit(clauses, rels, pred, found["mask"], target_mask)
        edited, info = edit_object(image, found["mask"], edit)
    except (ObjectNotFound, ValueError) as error:      # the photo stays untouched, the user is told why
        return {"image": image, "report": "\n".join(filter(None, [note, str(error)])), "marks": marks, "mode": "object"}
    h, w = image.shape[:2]
    lines = _head(rep, sentence, clauses, probs, note)
    lines += [f"Edited: {label} (found with score {found['score']:.2f})",
              f"Translation: tx = {info['tx']:+.0f} px, ty = {info['ty']:+.0f} px   (+x right, +y down)",
              f"Rotation:    {info['deg']:+.1f}°   (+ = counter-clockwise)",
              f"Scale:       ×{info['s']:.2f}",
              f"Image size:  {w} × {h}"]
    if "matrix" in info:                               # missing when nothing was left to edit
        lines += [f"Affine matrix (2×3), turning and sizing about the object's centre "
                  f"({info['centre'][0]:.0f}, {info['centre'][1]:.0f}):",
                  np.array2string(info["matrix"], precision=2, suppress_small=True)]
    lines += list(info["warnings"])
    return {"image": edited, "report": "\n".join(lines), "marks": marks, "mode": "object", "object": label,
            "params": {"tx": info["tx"], "ty": info["ty"], "deg": info["deg"], "s": info["s"], "warnings": info["warnings"]}}