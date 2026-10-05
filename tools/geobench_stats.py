import numpy as np
import pandas as pd
from datasets import load_dataset

ds = load_dataset("CIawevy/GeoBench", "2d", cache_dir="data/hf_cache")["data"]
ds = ds.select_columns(["edit_prompt", "edit_param"])   # skip the images: much faster

rows = []
format_problems = 0
for ex in ds:
    tx, ty, tz, rx, ry, rz, sx, sy, sz = ex["edit_param"]   # type: ignore
    prompt = ex["edit_prompt"].lower().strip(" .")          # type: ignore

    # 1. Format check: entries we don't use must hold their "do nothing" values
    if tz != 0 or rx != 0 or ry != 0 or sz != 1 or not np.isclose(sx, sy):
        format_problems += 1

    moved   = tx != 0 or ty != 0
    rotated = rz != 0
    scaled  = sx != 1

    if moved + rotated + scaled > 1:
        kind, size = "multi", 0.0
    elif moved:
        kind, size = "translate", np.hypot(tx, ty) / 512   # distance as fraction of width
    elif rotated:
        kind, size = "rotate", abs(rz)                      # degrees
    elif scaled:
        kind, size = "scale", abs(np.log(sx))               # |log s|
    else:
        kind, size = "none", 0.0

    rows.append({"kind": kind, "adverb": prompt.split()[-1], "size": size,
                 "ccw": "counterclockwise" in prompt, "rz": rz})

df = pd.DataFrame(rows)
print("rows:", len(df), "| format problems:", format_problems)
print(df["kind"].value_counts(), "\n")

# 2. Sign check
rot = df[df["kind"] == "rotate"]
print("counterclockwise with rz < 0:", (rot["ccw"] & (rot["rz"] < 0)).sum())
print("counterclockwise with rz > 0:", (rot["ccw"] & (rot["rz"] > 0)).sum(), "\n")

# 3. Range of sizes per adverb
for kind in ["translate", "rotate", "scale"]:
    table = df[df["kind"] == kind].groupby("adverb")["size"].agg(["count", "min", "median", "max"])
    print(f"== {kind} ==")
    print(table.sort_values("median").round(3), "\n")