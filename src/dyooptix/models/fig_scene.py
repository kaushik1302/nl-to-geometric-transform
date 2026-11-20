"""Figure for the scene-derived edit: one sentence, four gaps; predicted vs exact translation and the edited image."""
import cv2
import matplotlib.pyplot as plt

from dyooptix.data.shapes import draw, expected_touch
from dyooptix.geom.grounding import find_object
from dyooptix.geom.object_edit import edit_object
from dyooptix.models.breakdown import load_model
from dyooptix.models.eval_scene import REP, text_preds
from dyooptix.models.scene import read, scene_edit

SENTENCE = "move the circle right until it touches the square"
GAPS = [40, 100, 180, 260]                 # px between the circle and the square
CIRCLE = (100, 256, 40)                    # centre x, centre y, radius
SIDE = 80                                  # the square's side


def main():
    model, mean, std = load_model(REP)
    clauses, rels, mover, texts = read(SENTENCE)
    preds = text_preds(texts, model, mean, std)                    # same sentence, so one prediction for all four
    fig, axes = plt.subplots(2, len(GAPS), figsize=(4 * len(GAPS), 4.2))
    cx, cy, r = CIRCLE
    band = slice(cy - 90, cy + 90)                                 # only the strip with the objects
    for k, gap in enumerate(GAPS):
        x0 = cx + r + gap
        square = (x0, cy - SIDE // 2, x0 + SIDE - 1, cy + SIDE // 2 - 1)
        image, _, _ = draw(CIRCLE, square)
        circle = find_object(image, "circle")
        target = find_object(image, "square", avoid=circle["box"])
        edit = scene_edit(clauses, rels, preds, circle["mask"], target["mask"])
        out, info = edit_object(image, circle["mask"], edit)
        exact = expected_touch("right", CIRCLE, square)
        shown = out.copy()
        cv2.circle(shown, (round(cx + exact), cy), r, (0, 160, 0), 2)   # where the circle should end up

        axes[0, k].imshow(image[band])
        axes[0, k].set_title(f"gap {gap} px", fontsize=13)
        axes[1, k].imshow(shown[band])
        axes[1, k].set_title(f"predicted +{info['tx']:.0f} px, exact +{exact:.0f} px", fontsize=12)
        for ax in axes[:, k]:
            ax.axis("off")
    fig.suptitle(f'"{SENTENCE}"   (Grounded-SAM masks; green outline = exact end position)', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig("outputs/fig_scene_gaps.png", dpi=150, bbox_inches="tight")
    print("saved outputs/fig_scene_gaps.png")


if __name__ == "__main__":
    main()