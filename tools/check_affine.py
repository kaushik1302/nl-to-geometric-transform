import numpy as np
import cv2
from dyooptix.geom.affine import build_affine

# 1. "Do nothing" must give the identity
M0 = build_affine({"tx": 0, "ty": 0, "cos": 1, "sin": 0, "log_s": 0}, (100, 80), 256)
print("identity:\n", M0)

# 2. Must match OpenCV when there's no translation
deg, s, c = 30, 1.5, (100, 80)
mine = build_affine({"tx": 0, "ty": 0,
                     "cos": np.cos(np.radians(deg)), "sin": np.sin(np.radians(deg)),
                     "log_s": np.log(s)}, c, 256)
ref = cv2.getRotationMatrix2D(c, deg, s)
print("mine:\n", mine)
print("opencv:\n", ref)
print("match:", np.allclose(mine, ref))