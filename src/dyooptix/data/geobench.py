"""Convert GeoBench 2D into a test file in our format: data/splits/geobench.jsonl"""
import json
import math

from datasets import load_dataset

from dyooptix.data import vocab

# GeoBench's adverbs, grouped into the three levels measured by tools/geobench_stats.py
LEVEL_OF = {
    "gently": "small", "mildly": "small", "lightly": "small", "slightly": "small",
    "appreciably": "medium", "markedly": "medium", "moderately": "medium",
    "heavily": "large", "strongly": "large", "significantly": "large", "intensely": "large",
}
TRAINING_ADVERBS = {word for words in vocab.ADVERBS.values() for word in words}

ds = load_dataset("CIawevy/GeoBench", "2d", cache_dir="data/hf_cache")["data"]
ds = ds.select_columns(["edit_prompt", "edit_param"])   # text only: skip the images

with open("data/splits/geobench.jsonl", "w", encoding="utf-8") as f:
    for i, ex in enumerate(ds):
        tx, ty, tz, rx, ry, rz, sx, sy, sz = ex["edit_param"] # type: ignore
        theta = math.radians(-rz)        # GeoBench: clockwise positive; ours: counter-clockwise
        target = [tx / 512, ty / 512, math.cos(theta), math.sin(theta), math.log(sx)]
        presence = [int(tx != 0 or ty != 0), int(rz != 0), int(sx != 1)]
        category = ["translate", "rotate", "scale"][presence.index(1)]
        instruction = ex["edit_prompt"].lower().strip(" .") # type: ignore
        adverb = instruction.split()[-1]

        row = {
            "instruction": instruction,
            "target": [round(v, 4) for v in target],
            "presence": presence,
            "category": category,
            "edits": [category],
            "levels": [LEVEL_OF[adverb]],
            "adverb_seen": adverb in TRAINING_ADVERBS,
            "geobench_index": i,
        }
        f.write(json.dumps(row) + "\n")

print("wrote", len(ds), "rows")