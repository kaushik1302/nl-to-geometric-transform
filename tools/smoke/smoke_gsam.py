import sys
import time

import cv2
import numpy as np
from PIL import Image

from dyooptix.geom.grounding import ObjectNotFound, find_object, load, outline

photo, name = sys.argv[1], sys.argv[2]
image = np.array(Image.open(photo).convert("RGB"))

t = time.time()
load()
print(f"load models: {time.time() - t:.1f} s")

for attempt in ("first call", "same call again"):
    t = time.time()
    found = find_object(image, name)
    print(f"{attempt}: {time.time() - t:.1f} s   box {[round(v) for v in found['box']]}   score {found['score']:.2f}")

cv2.imwrite("mask_check.png", found["mask"])
print(f"outline: {len(outline(found['mask']))} points, mask {found['mask'].mean() / 255:.1%} of the photo")

try:
    find_object(image, "toaster")
except ObjectNotFound as error:
    print("missing object ->", error)