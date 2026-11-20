"""Edit one object in a photo: clamp the edit, fill the hole it leaves, paste the moved object with a soft edge."""
import cv2
import numpy as np

from dyooptix.geom.affine import build_affine
from dyooptix.geom.clamp import clamp
from dyooptix.geom.grounding import outline
from dyooptix.geom.inpaint import fill

SHRINK = 1       # object mask eroded by this many pixels before softening, so the soft edge stays on the object
SOFT = 1.5       # blur (sigma, in pixels) that softens the pasted object's edge


def edit_object(image, mask, target):
    """image: RGB uint8 (H, W, 3); mask: object pixels (white);
    target: [tx, ty, cos, sin, log s] with tx, ty as fractions of the photo's width.
    Returns (edited image, info) where info holds the clamped edit and the warnings."""
    H, W = mask.shape
    ys, xs = np.nonzero(mask > 127)
    centre = (xs.mean(), ys.mean())                                # pivot: centroid of the object's pixels

    deg = np.degrees(np.arctan2(target[3], target[2]))
    tx, ty, deg, s, warnings = clamp(outline(mask), centre, (W, H),
                                     target[0] * W, target[1] * W, deg, np.exp(target[4]))
    info = {"tx": tx, "ty": ty, "deg": deg, "s": s, "warnings": warnings, "centre": centre}
    if abs(tx) < 0.5 and abs(ty) < 0.5 and abs(deg) < 0.1 and abs(s - 1) < 1e-3:
        return image, info                                         # nothing to do: the photo stays untouched

    a = np.radians(deg)
    M = build_affine({"tx": tx / W, "ty": ty / W, "cos": np.cos(a), "sin": np.sin(a), "log_s": np.log(s)}, centre, W)
    info["matrix"] = M                                             # shown in the demo's report

    alpha = cv2.erode((mask > 127).astype(np.float32), np.ones((3, 3), np.uint8), iterations=SHRINK)
    alpha = cv2.GaussianBlur(alpha, (0, 0), SOFT)                  # 1 inside the object, fading to 0 at its edge
    moved = cv2.warpAffine(image, M, (W, H), flags=cv2.INTER_CUBIC).astype(np.float32)
    alpha = cv2.warpAffine(alpha, M, (W, H), flags=cv2.INTER_LINEAR)[..., None]

    background = fill(image, mask).astype(np.float32)              # the photo with the object removed
    out = alpha * moved + (1 - alpha) * background
    return np.clip(out + 0.5, 0, 255).astype(np.uint8), info