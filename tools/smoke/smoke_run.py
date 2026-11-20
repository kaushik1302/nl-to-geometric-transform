import sys
import time

import cv2
import numpy as np
from PIL import Image

from dyooptix.models.pipeline import run

photo, sentence = sys.argv[1], sys.argv[2]
object_name = sys.argv[3] if len(sys.argv) > 3 else ""
image = np.array(Image.open(photo).convert("RGB"))

t = time.time()
result = run(image, sentence, object_name=object_name)
print(f"{time.time() - t:.1f} s ({result['mode']} edit; the first run includes loading the models)")
print(result["report"])
print("marks:", result["marks"])
cv2.imwrite("run_check.png", cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR))