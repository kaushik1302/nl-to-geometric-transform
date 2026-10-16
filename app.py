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

EDITS = ["translate", "rotate", "scale"]
MODELS = {rep: load_model(rep) for rep in ["hand", "minilm"]}  # loaded once, at start-up
MINILM = SentenceTransformer("all-MiniLM-L6-v2")


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
    if rep == "hand":
        return np.array([hand_features(sentence)], dtype=np.float32)
    return MINILM.encode([sentence])


def edit(instruction, image, rep):
    if image is None:
        image = sample_image()
    sentence = instruction.lower().strip().rstrip(".!?")
    if not sentence:
        return image, "Type an instruction."

    model, mean, std = MODELS[rep]
    x = standardise(encode(sentence, rep), mean, std)
    pred, _ = predict(model, x)                      # gated [tx, ty, cos, sin, log s]
    with torch.no_grad():
        probs = torch.sigmoid(model(x)[0])[0].numpy()  # presence probabilities, for display

    target = pred[0]
    p = to_params(target)
    h, w = image.shape[:2]
    angle = np.degrees(np.arctan2(p["sin"], p["cos"]))
    M = build_affine(p, (w / 2, h / 2), w)

    lines = ["Detected edits (probability):"]
    lines += [f"  {name:9}  {prob:.2f}  {'✓' if prob > 0.5 else ''}" for name, prob in zip(EDITS, probs)]
    lines += [f"Translation: tx = {p['tx'] * w:+.0f} px, ty = {p['ty'] * w:+.0f} px   (+x right, +y down)",
              f"Rotation:    {angle:+.1f}°   (+ = counter-clockwise)",
              f"Scale:       ×{np.exp(p['log_s']):.2f}",
              "Affine matrix (2×3):",
              np.array2string(M, precision=2, suppress_small=True)]
    return warp(image, target, cv2.INTER_LINEAR), "\n".join(lines)


demo = gr.Interface(
    fn=edit,
    inputs=[
        gr.Textbox(label="Edit instruction", placeholder="e.g. rotate it slightly clockwise and move it to the left"),
        gr.Image(type="numpy", label="Image (leave empty to use the test card)"),
        gr.Radio(["minilm", "hand"], value="minilm", label="Text representation"),
    ],
    outputs=[gr.Image(label="Edited image"), gr.Textbox(label="Predicted parameters", lines=10)],
    examples=[
        ["rotate the shape by 30 degrees clockwise", None, "hand"],
        ["move it slightly to the left and enlarge it a lot", None, "minilm"],
        ["shrink the picture by a factor of 2", None, "hand"],
        ["spin it counter-clockwise significantly", None, "hand"],
        ["drag the box upward mildly", None, "minilm"],
    ],
    title="DyoOptix: natural-language geometric edits",
    description="Type an edit instruction. The model predicts translation, rotation and scale, "
                "and applies them to the whole image about its centre.",
)

if __name__ == "__main__":
    demo.launch()