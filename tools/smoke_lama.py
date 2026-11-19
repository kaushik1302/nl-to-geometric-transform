import os
import sys
import time

CACHE = os.environ.get("DYOOPTIX_CACHE", "data/hf_cache")
os.environ["TORCH_HOME"] = CACHE             # must be set before the import below, so the weights go there

import cv2
import numpy as np
from PIL import Image
from simple_lama_inpainting import SimpleLama

photo, mask_file = sys.argv[1], sys.argv[2]
size = int(sys.argv[3]) if len(sys.argv) > 3 else 512         # long side LaMa works at
margin = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5     # extra context, as a fraction of the box size

image = cv2.cvtColor(cv2.imread(photo), cv2.COLOR_BGR2RGB)
mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
mask = cv2.dilate((mask > 127).astype(np.uint8) * 255, np.ones((15, 15), np.uint8))  # grow 7 px: covers edge halo
H, W = mask.shape

ys, xs = np.nonzero(mask)                                      # crop: box around the mask, plus margin
pad_x, pad_y = int(margin * (xs.max() - xs.min())), int(margin * (ys.max() - ys.min()))
x0, x1 = max(xs.min() - pad_x, 0), min(xs.max() + pad_x + 1, W)
y0, y1 = max(ys.min() - pad_y, 0), min(ys.max() + pad_y + 1, H)
crop, crop_mask = image[y0:y1, x0:x1], mask[y0:y1, x0:x1]

scale = min(1.0, size / max(crop.shape[:2]))                   # shrink only, never enlarge
h, w = round(crop.shape[0] * scale), round(crop.shape[1] * scale)
small = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)
small_mask = cv2.resize(crop_mask, (w, h), interpolation=cv2.INTER_NEAREST)
print(f"crop {x1 - x0} x {y1 - y0} -> LaMa works at {w} x {h}  (scale {scale:.2f})")

lama = SimpleLama()
t = time.time()
piece = np.array(lama(Image.fromarray(small), Image.fromarray(small_mask)))[:h, :w]   # LaMa pads to a multiple of 8
print(f"LaMa {time.time() - t:.1f} s")

piece = cv2.resize(piece, (x1 - x0, y1 - y0), interpolation=cv2.INTER_CUBIC)         # back to the crop's real size
out = image.copy()
inside = crop_mask > 0
out[y0:y1, x0:x1][inside] = piece[inside]                      # only the masked pixels change
cv2.imwrite("lama_small_out.png", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))