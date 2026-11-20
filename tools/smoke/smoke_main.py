import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from dyooptix.geom.grounding import FLOOR, load

dino_proc, dino, _, _ = load()

for path in sys.argv[1:]:
    image = np.array(Image.open(path).convert("RGB"))
    H, W = image.shape[:2]
    inputs = dino_proc(images=Image.fromarray(image), text="object.", return_tensors="pt",
                       size={"shortest_edge": 480, "longest_edge": 640})
    with torch.no_grad():
        out = dino(**inputs)
    found = dino_proc.post_process_grounded_object_detection(
        out, inputs.input_ids, threshold=FLOOR, text_threshold=0.25, target_sizes=[(H, W)])[0]

    shown = image.copy()
    print(path)
    for k, (score, box) in enumerate(zip(found["scores"].tolist(), found["boxes"].tolist())):
        area = (box[2] - box[0]) * (box[3] - box[1]) / (W * H)
        print(f"  {k}: score {score:.2f}  area {area:.0%}  box {[round(v) for v in box]}")
        x0, y0, x1, y1 = (int(v) for v in box)
        cv2.rectangle(shown, (x0, y0), (x1, y1), (255, 0, 0), 3)
        cv2.putText(shown, str(k), (x0 + 5, y0 + 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
    if not len(found["scores"]):
        print("  no boxes")
    cv2.imwrite(f"main_check_{Path(path).stem}.png", cv2.cvtColor(shown, cv2.COLOR_RGB2BGR))