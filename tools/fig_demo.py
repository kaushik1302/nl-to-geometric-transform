"""Poster figure: for each example, the input photo, the instruction with its predicted parameters, and the edited result."""
import re
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # lets this script import app.py (the test card)

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from app import sample_image
from dyooptix.models.pipeline import run

EXAMPLES = [                                                    # (photo, instruction); None = the test card
    ("samples/mugbottle.jpeg", "rotate the image by 37 degrees clockwise"),
    ("samples/cup.jpg", "move the cup 300 px to the right"),
    ("samples/table4.jpeg", "shrink the image to 0.7x and turn it slightly counter-clockwise"),
    ("samples/teapot.jpeg", "rotate it 25 degrees clockwise and make it twice as big"),
    ("samples/table4.jpeg", "move the mug until it touches the bottle"),
    (None, "move the circle beside the square"),
]
SHOWN = ("Translation", "Rotation", "Scale", "⚠")               # the report lines worth showing
NOTE = re.compile(r"\s*\(\+[^)]*\)")                             # "(+x right, +y down)", "(+ = counter-clockwise)"
WIDTH = 38                                                      # characters per line in the middle panel

fig, axes = plt.subplots(len(EXAMPLES), 3, figsize=(12, 3.4 * len(EXAMPLES)),
                         gridspec_kw={"width_ratios": [1, 0.7, 1]})
for row, (photo, sentence) in zip(axes, EXAMPLES):
    image = sample_image() if photo is None else np.array(Image.open(photo).convert("RGB"))
    result = run(image, sentence)
    params = [" ".join(NOTE.sub("", line).split()) for line in result["report"].splitlines() if line.startswith(SHOWN)]
    text = (textwrap.fill(f'"{sentence}"', WIDTH) + "\n\n"
            + "\n".join(textwrap.fill(line, WIDTH, subsequent_indent="  ") for line in params))
    row[0].imshow(image)
    row[2].imshow(result["image"])
    row[1].text(0.5, 0.5, text, transform=row[1].transAxes, ha="center", va="center", family="monospace", fontsize=10)
    for ax in row:
        ax.axis("off")
    print(sentence)

for ax, title in zip(axes[0], ("Input", "Instruction and predicted parameters", "Edited result")):
    ax.set_title(title, fontsize=12, fontweight="bold")
fig.text(0.5, 0.0, "tx, ty in pixels of the photo (+x right, +y down); rotation in degrees (+ = counter-clockwise); "
                   "scale as a factor. Rows 2 and 4-6 edit one object, found from the text; rows 1 and 3 edit the whole image.",
         ha="center", va="top", fontsize=9, wrap=True)
fig.tight_layout()
fig.savefig("outputs/fig_demo_examples.png", dpi=130, bbox_inches="tight")