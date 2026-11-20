import sys, platform
print(f"Python  {sys.version.split()[0]}  ({platform.system()})")

import numpy as np;    print("numpy   ", np.__version__)
import cv2;            print("opencv  ", cv2.__version__)
import sklearn;        print("sklearn ", sklearn.__version__)
import matplotlib;     print("mpl     ", matplotlib.__version__)
import torch;          print("torch   ", torch.__version__)

print("\n-- opencv warp --")
img = np.zeros((64, 64), dtype=np.uint8)
img[20:40, 20:40] = 255
M = cv2.getRotationMatrix2D((32.0, 32.0), 45.0, 1.0)
out = cv2.warpAffine(img, M, (64, 64), flags=cv2.INTER_NEAREST)
print("warp ok, nonzero px:", int((out > 0).sum()))

print("\n-- encoders (first run downloads weights) --")
S = ["move the square slightly to the left"]

import open_clip
model, _, _ = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k")
tok = open_clip.get_tokenizer("ViT-B-32")
with torch.no_grad():
    clip_vec = model.encode_text(tok(S))  # type: ignore[attr-defined]
print("CLIP    ", tuple(clip_vec.shape))

from sentence_transformers import SentenceTransformer
mini_vec = SentenceTransformer("all-MiniLM-L6-v2").encode(S)
print("MiniLM  ", mini_vec.shape)

print("\nAll good.")