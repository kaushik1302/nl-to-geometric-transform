import sys
import time

import cv2
import numpy as np

from dyooptix.geom.grounding import find_object
from dyooptix.geom.object_edit import edit_object

photo, name = sys.argv[1], sys.argv[2]
tx_px, ty_px, deg, scale = (float(v) for v in sys.argv[3:7])   # the edit, typed directly (no text model here)

image = cv2.cvtColor(cv2.imread(photo), cv2.COLOR_BGR2RGB)
W = image.shape[1]
a = np.radians(deg)
target = [tx_px / W, ty_px / W, np.cos(a), np.sin(a), np.log(scale)]

t = time.time()
mask = find_object(image, name)["mask"]
print(f"find {name}: {time.time() - t:.1f} s")

t = time.time()
out, info = edit_object(image, mask, target)
print(f"edit: {time.time() - t:.1f} s")
print(f"asked:   move ({tx_px:+.0f}, {ty_px:+.0f}) px, rotate {deg:+.0f}°, scale ×{scale:.2f}")
print(f"applied: move ({info['tx']:+.0f}, {info['ty']:+.0f}) px, rotate {info['deg']:+.0f}°, scale ×{info['s']:.2f}")
for w in info["warnings"]:
    print(w)
cv2.imwrite("edit_out.png", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))