import sys
import time

import cv2

from dyooptix.geom.inpaint import fill, load

photo, mask_file = sys.argv[1], sys.argv[2]
image = cv2.cvtColor(cv2.imread(photo), cv2.COLOR_BGR2RGB) # type: ignore
mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE) # type: ignore

t = time.time()
load()
print(f"load LaMa: {time.time() - t:.1f} s")

t = time.time()
out = fill(image, mask)
print(f"fill: {time.time() - t:.1f} s")
cv2.imwrite("fill_check.png", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))