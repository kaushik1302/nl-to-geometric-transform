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
image = cv2.cvtColor(cv2.imread(photo), cv2.COLOR_BGR2RGB)
mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
mask = cv2.dilate((mask > 127).astype(np.uint8) * 255, np.ones((15, 15), np.uint8))  # grow 7 px: covers edge halo
h, w = mask.shape

t = time.time()
lama = SimpleLama()
print(f"load {time.time() - t:.1f} s")

t = time.time()
filled = np.array(lama(Image.fromarray(image), Image.fromarray(mask)))[:h, :w]
print(f"LaMa {time.time() - t:.1f} s")

telea = cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)
cv2.imwrite("lama_out.png", cv2.cvtColor(filled, cv2.COLOR_RGB2BGR))
cv2.imwrite("telea_out.png", cv2.cvtColor(telea, cv2.COLOR_RGB2BGR))