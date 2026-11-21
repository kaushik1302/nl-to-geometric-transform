import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))  # lets app.py import dyooptix without installing the package

import cv2
import gradio as gr
import numpy as np

from dyooptix.geom import grounding, inpaint
from dyooptix.models.pipeline import REPS, models, run


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


def draw(image, marks):
    """The photo with every candidate object boxed and labelled."""
    shown = image.copy()
    thick = max(2, round(max(shown.shape[:2]) / 250))
    for m in marks:
        x0, y0, x1, y1 = (int(v) for v in m["box"])
        cv2.rectangle(shown, (x0, y0), (x1, y1), (255, 140, 0), thick)
        cv2.putText(shown, m["label"], (x0 + thick, y0 + 12 * thick), cv2.FONT_HERSHEY_SIMPLEX,
                    thick / 3, (255, 140, 0), thick)
    return shown


def edit(instruction, image, object_name="", rep=REPS[0], use_split=True):
    if image is None:
        image = sample_image()
    result = run(image, instruction, rep, use_split, object_name or "")
    considered = draw(image, result["marks"]) if result["marks"] else None   # only when the main object was a tie
    return result["image"], considered, result["report"]


models()                       # everything is loaded once, at start-up
grounding.load()
inpaint.load()

demo = gr.Interface(
    fn=edit,
    inputs=[
        gr.Textbox(label="Edit instruction", placeholder="e.g. move the cup 100 px to the right and rotate it 20 degrees clockwise"),
        gr.Image(type="numpy", label="Image (leave empty to use the test card)"),
        gr.Textbox(label="Object (optional)", placeholder="the object to edit; leave empty to use the one in the instruction"),
    ],
    additional_inputs=[                                  # hidden in a collapsed panel; defaults are used otherwise
        gr.Radio(REPS, value=REPS[0], label="Text representation"),
        gr.Checkbox(value=True, label="Split into clauses (and / then / commas)"),
    ],
    additional_inputs_accordion=gr.Accordion("Advanced (for comparison)", open=False),
    outputs=[gr.Image(label="Edited image"),
             gr.Image(label="What the app considered (shown when several objects look equally main)"),
             gr.Textbox(label="Predicted parameters", lines=14)],
    examples=[
        ["rotate the image by 90 degrees clockwise", None, "", "hand+minilm", True],
        ["move the image 100 px to the right", None, "", "hand+minilm", True],
        ["scale the image to 0.5x", None, "", "hand+minilm", True],
        ["shrink the image to 0.7x and turn it slightly counter-clockwise", None, "", "hand+minilm", True],
        ["move the square slightly to the left and enlarge it a lot", None, "", "hand+minilm", True],
        ["move the circle 100 px left, then rotate it 30 degrees clockwise and make it twice as big", None, "", "hand+minilm", True],
        ["move the circle until it touches the square", None, "", "hand+minilm", True],
        ["move the circle beside the square", None, "", "hand+minilm", True],
    ],
    title="DyoOptix: natural-language geometric edits",
    description="Type an edit instruction and optionally upload an image. The model predicts translation, rotation "
                "and scale. Name an object ('the cup') to edit only that object (the hole it leaves is filled in); "
                "say 'the image' to edit the whole image; name nothing and the main object is edited. "
                "Combined instructions are split into clauses, each predicted separately and then added up; "
                "exact numbers in the text (degrees, pixels, percent, factors) are used as written; "
                "'until it touches', 'beside' and 'until the left edge' are measured from the photo itself. "
                "The best overall model (hand features + MiniLM) is used automatically; "
                "Advanced lets you compare other text representations and turn splitting off.",
)

if __name__ == "__main__":
    demo.launch(share="--share" in sys.argv)   # python app.py --share: temporary public link