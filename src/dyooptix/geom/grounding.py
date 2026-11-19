"""Text -> object mask: Grounding DINO turns the word into a box, SAM turns the box into a mask.
Models load once; results are cached per (photo, word)."""
import hashlib
import os
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image

CACHE = os.environ.get("DYOOPTIX_CACHE", "data/hf_cache")
DINO_ID = "IDEA-Research/grounding-dino-tiny"
SAM_ID = "facebook/sam-vit-base"
MIN_SCORE = 0.30          # below this, DINO is not sure the object is in the photo
FLOOR = 0.05              # lowest score kept, only so the error message can report the best rejected match

_found = {}               # (photo fingerprint, object word) -> result of _ground


class ObjectNotFound(Exception):
    pass


@lru_cache(maxsize=None)
def load():
    """Load both models. Cached: only the first call is slow."""
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor
    return (AutoProcessor.from_pretrained(DINO_ID, cache_dir=CACHE),
            AutoModelForZeroShotObjectDetection.from_pretrained(DINO_ID, cache_dir=CACHE),
            SamProcessor.from_pretrained(SAM_ID, cache_dir=CACHE),
            SamModel.from_pretrained(SAM_ID, cache_dir=CACHE))


def find_object(image, name):
    """image: RGB uint8 array (H, W, 3). Returns {"mask": 0/255 uint8 (H, W), "box": [x0, y0, x1, y1], "score": float}.
    Raises ObjectNotFound when the word is not in the photo."""
    name = name.lower().strip()
    key = (hashlib.md5(image.tobytes()).hexdigest(), name)
    if key not in _found:
        _found[key] = _ground(image, name)
    return _found[key]


def _ground(image, name):
    import torch
    dino_proc, dino, sam_proc, sam = load()
    photo = Image.fromarray(image)
    H, W = image.shape[:2]

    # 1) text -> box, on a small copy of the photo (fast); boxes come back in the original photo's pixels
    inputs = dino_proc(images=photo, text=name + ".", return_tensors="pt",
                       size={"shortest_edge": 480, "longest_edge": 640})
    with torch.no_grad():
        out = dino(**inputs)
    found = dino_proc.post_process_grounded_object_detection(
        out, inputs.input_ids, threshold=FLOOR, text_threshold=0.25, target_sizes=[(H, W)])[0]
    scores = found["scores"]
    if len(scores) == 0 or scores.max() < MIN_SCORE:
        best = float(scores.max()) if len(scores) else 0.0
        raise ObjectNotFound(f"Couldn't find '{name}' in the photo (best match {best:.2f}, needs {MIN_SCORE:.2f}). "
                             "Try another word in the Object box, or leave the object out to edit the whole image.")
    i = scores.argmax()
    box = found["boxes"][i].tolist()

    # 2) box -> mask; SAM proposes three masks, keep the one it scores highest
    inputs = sam_proc(photo, input_boxes=[[box]], return_tensors="pt")
    with torch.no_grad():
        out = sam(**inputs)
    masks = sam_proc.image_processor.post_process_masks(
        out.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"])[0]
    mask = masks[0, out.iou_scores[0, 0].argmax()].numpy().astype(np.uint8) * 255
    return {"mask": mask, "box": box, "score": float(scores[i])}


def outline(mask):
    """Convex hull of the mask's largest blob, as (N, 2) float points (x, y): the shape the clamp works with."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest = max(contours, key=cv2.contourArea)
    return cv2.convexHull(biggest).reshape(-1, 2).astype(float)