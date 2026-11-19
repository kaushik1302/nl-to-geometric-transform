"""Fill the hole an object leaves behind with LaMa, on a crop around the hole shrunk to at most 512 px."""
import os
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image

CACHE = os.environ.get("DYOOPTIX_CACHE", "data/hf_cache")
os.environ["TORCH_HOME"] = CACHE          # LaMa's weights download here
SIZE = 512                                # longest side LaMa works at
MARGIN = 0.5                              # extra context around the hole, as a fraction of its box
GROW = 15                                 # mask is grown with a 15 x 15 square: about 7 px on each side


@lru_cache(maxsize=None)
def load():
    """Load LaMa. Cached: only the first call is slow."""
    from simple_lama_inpainting import SimpleLama
    return SimpleLama()


def fill(image, mask):
    """image: RGB uint8 (H, W, 3); mask: object pixels (white). Returns a copy with that area filled in."""
    H, W = mask.shape
    mask = cv2.dilate((mask > 127).astype(np.uint8) * 255, np.ones((GROW, GROW), np.uint8))
    ys, xs = np.nonzero(mask)                                      # crop: box around the hole, plus margin
    pad_x, pad_y = int(MARGIN * (xs.max() - xs.min())), int(MARGIN * (ys.max() - ys.min()))
    x0, x1 = max(xs.min() - pad_x, 0), min(xs.max() + pad_x + 1, W)
    y0, y1 = max(ys.min() - pad_y, 0), min(ys.max() + pad_y + 1, H)
    crop, crop_mask = image[y0:y1, x0:x1], mask[y0:y1, x0:x1]

    scale = min(1.0, SIZE / max(crop.shape[:2]))                   # shrink only, never enlarge
    h, w = round(crop.shape[0] * scale), round(crop.shape[1] * scale)
    small = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(crop_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    piece = np.array(load()(Image.fromarray(small), Image.fromarray(small_mask)))[:h, :w]   # LaMa pads to a multiple of 8
    piece = cv2.resize(piece, (x1 - x0, y1 - y0), interpolation=cv2.INTER_CUBIC)

    out = image.copy()
    inside = crop_mask > 0
    out[y0:y1, x0:x1][inside] = piece[inside]                      # only the hole's pixels change
    return out