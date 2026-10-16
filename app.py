import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))  # lets app.py import dyooptix without installing the package

import cv2
import gradio as gr
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from dyooptix.geom.affine import build_affine
from dyooptix.geom.warp import warp
from dyooptix.models.breakdown import load_model
from dyooptix.models.encode import hand_features
from dyooptix.models.metrics import to_params
from dyooptix.models.mlp import predict, standardise

REPS = ["hand+minilm", "hand", "minilm"]   # the first one is the default
EDITS = ["translate", "rotate", "scale"]
REFERENCE_WIDTH = 512                      # explicit pixels in training refer to a 512-wide image
PIXELS = re.compile(r"\d\s*(?:pixels|px)\b")

# Spelling variants of the same word -> the spelling used in training (demo input cleaning only)
SPELLINGS = {
    "clock-wise": "clockwise", "clock wise": "clockwise",
    "anti-clockwise": "anticlockwise", "anti clockwise": "anticlockwise",
    "counterclockwise": "counter-clockwise", "counter clockwise": "counter-clockwise",
}

MODELS = {rep: load_model(rep) for rep in REPS}  # loaded once, at start-up
MINILM = SentenceTransformer("all-MiniLM-L6-v2")


def normalise(sentence):
    for variant, canonical in SPELLINGS.items():
        sentence = sentence.replace(variant, canonical)
    return sentence


def sample_image(size=512):
    """Test card: grid, square, upward arrow, circle and 'TOP', so every edit is easy to see."""
    image = np.full((size, size, 3), 255, dtype=np.uint8)
    for k in range(0, size, 64):
        cv2.line(image, (k, 0), (k, size), (220, 220, 220), 1)
        cv2.line(image, (0, k), (size, k), (220, 220, 220), 1)
    cv2.rectangle(image, (176, 176), (336, 336), (40, 90, 200), -1)
    cv2.arrowedLine(image, (256, 310), (256, 200), (255, 255, 255), 8, tipLength=0.3)
    cv2.circle(image, (400, 110), 40, (220, 60, 60), -1)
    cv2.putText(image, "TOP", (215, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    return image


def encode(sentence, rep):
    """Same features as in training; "hand+minilm" stacks hand then MiniLM, like load_features."""
    parts = []
    for part in rep.split("+"):
        if part == "hand":
            parts.append(np.array([hand_features(sentence)], dtype=np.float32))
        else:
            parts.append(MINILM.encode([sentence]))
    return np.hstack(parts)


def edit(instruction, image, rep):
    if image is None:
        image = sample_image()
    sentence = normalise(instruction.lower().strip().rstrip(".!?"))
    if not sentence:
        return image, "Type an instruction."

    model, mean, std = MODELS[rep]
    x = standardise(encode(sentence, rep), mean, std)
    pred, _ = predict(model, x)                        # gated [tx, ty, cos, sin, log s]
    with torch.no_grad():
        probs = torch.sigmoid(model(x)[0])[0].numpy()  # presence probabilities, for display

    target = pred[0].copy()
    h, w = image.shape[:2]
    explicit_px = bool(PIXELS.search(sentence))
    if explicit_px:
        # tx, ty are fractions of a 512-px width; "50 px" must mean 50 real pixels of this image
        target[0] *= REFERENCE_WIDTH / w
        target[1] *= REFERENCE_WIDTH / w

    p = to_params(target)
    angle = np.degrees(np.arctan2(p["sin"], p["cos"]))
    M = build_affine(p, (w / 2, h / 2), w)

    lines = [f"Model: {rep}", f"Text used: {sentence}", "Detected edits (probability):"]
    lines += [f"  {name:9}  {prob:.2f}  {'✓' if prob > 0.5 else ''}" for name, prob in zip(EDITS, probs)]
    lines += [f"Translation: tx = {p['tx'] * w:+.0f} px, ty = {p['ty'] * w:+.0f} px   (+x right, +y down)"
              + ("   [explicit pixels: real pixels of this image]" if explicit_px else ""),
              f"Rotation:    {angle:+.1f}°   (+ = counter-clockwise)",
              f"Scale:       ×{np.exp(p['log_s']):.2f}",
              f"Image size:  {w} × {h}",
              "Affine matrix (2×3):",
              np.array2string(M, precision=2, suppress_small=True)]
    return warp(image, target, cv2.INTER_LINEAR), "\n".join(lines)


demo = gr.Interface(
    fn=edit,
    inputs=[
        gr.Textbox(label="Edit instruction", placeholder="e.g. rotate it 90 degrees clockwise and shrink it to 0.7x"),
        gr.Image(type="numpy", label="Image (leave empty to use the test card)"),
        gr.Radio(REPS, value=REPS[0], label="Text representation"),
    ],
    outputs=[gr.Image(label="Edited image"), gr.Textbox(label="Predicted parameters", lines=13)],
    examples=[
        ["rotate the image by 90 degrees clockwise", None, "hand+minilm"],
        ["move it 100 px to the right", None, "hand"],
        ["scale it to 0.5x", None, "hand+minilm"],
        ["shrink it to 0.7x and turn it slightly counter-clockwise", None, "hand+minilm"],
        ["move it slightly to the left and enlarge it a lot", None, "hand"],
        ["drag the box upward mildly", None, "hand+minilm"],
    ],
    title="DyoOptix: natural-language geometric edits",
    description="Type an edit instruction. The model predicts translation, rotation and scale, "
                "and applies them to the whole image about its centre. "
                "hand+minilm is the default; hand is most precise with known words and exact numbers; "
                "minilm copes best with new wording on its own.",
)

if __name__ == "__main__":
    demo.launch(share="--share" in sys.argv)  # python app.py --share  ->  temporary public link