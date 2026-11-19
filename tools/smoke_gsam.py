import os
import sys, time, numpy as np, torch, cv2
from PIL import Image
from transformers import (AutoProcessor, AutoModelForZeroShotObjectDetection,
                          SamModel, SamProcessor)

CACHE = os.environ.get("DYOOPTIX_CACHE", "data/hf_cache")
path, name = sys.argv[1], sys.argv[2]
image = Image.open(path).convert("RGB")
W, H = image.size
print("torch threads:", torch.get_num_threads())

# Load once (in the demo this happens at start-up, not per edit)
t = time.time()
dino_id = "IDEA-Research/grounding-dino-tiny"
dino_proc = AutoProcessor.from_pretrained(dino_id, cache_dir=CACHE)
dino = AutoModelForZeroShotObjectDetection.from_pretrained(dino_id, cache_dir=CACHE)
sam = SamModel.from_pretrained("facebook/sam-vit-base", cache_dir=CACHE)
sam_proc = SamProcessor.from_pretrained("facebook/sam-vit-base", cache_dir=CACHE)
print(f"load models: {time.time() - t:.1f}s")

# 1) Grounding DINO on a small version of the photo: text -> box
t = time.time()
inp = dino_proc(images=image, text=name.lower() + ".", return_tensors="pt",
                size={"shortest_edge": 480, "longest_edge": 640})
with torch.no_grad():
    out = dino(**inp)
res = dino_proc.post_process_grounded_object_detection(
    out, inp.input_ids, threshold=0.3, text_threshold=0.25,
    target_sizes=[(H, W)])[0]          # boxes come back in the ORIGINAL photo's pixels
if len(res["boxes"]) == 0:
    sys.exit("object not found")
best_box = res["scores"].argmax()
box = res["boxes"][best_box].tolist()
print("box", [round(v) for v in box], "score", round(float(res["scores"][best_box]), 2))
print(f"DINO: {time.time() - t:.1f}s")

# 2) SAM on the full photo: box -> mask
t = time.time()
inp = sam_proc(image, input_boxes=[[box]], return_tensors="pt")
with torch.no_grad():
    out = sam(**inp)
masks = sam_proc.image_processor.post_process_masks(
    out.pred_masks, inp["original_sizes"], inp["reshaped_input_sizes"])[0]
best = out.iou_scores[0, 0].argmax()
mask = masks[0, best].numpy().astype(np.uint8) * 255
cv2.imwrite("mask_v2.png", mask)
print(f"saved mask_v2.png  ({mask.mean() / 255:.1%} of image)")
print(f"SAM: {time.time() - t:.1f}s")