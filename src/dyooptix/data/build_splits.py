"""Split generated data into train / val / test, plus a test set of unseen sentence patterns."""
import json
import os
import random

from dyooptix.data import vocab
from dyooptix.data.generate import make_example

rng = random.Random(42)

# 1. Generate 7,000 examples, skipping exact duplicates (same sentence and same numbers)
rows = []
seen = set()
for _ in range(7000):
    row = make_example(rng)
    key = (row["instruction"], tuple(row["target"]))
    if key not in seen:
        seen.add(key)
        rows.append(row)
print("duplicates removed:", 7000 - len(rows))

# 2. "can you ..." patterns go only to the unseen-template test set
unseen, rest = [], []
for r in rows:
    if r["template_id"] == vocab.HELD_OUT_TEMPLATES.get(r["category"]):
        unseen.append(r)
    else:
        rest.append(r)

# 3. Shuffle the rest and cut it 80 / 10 / 10
rng.shuffle(rest)
n = len(rest)
splits = {
    "train": rest[: int(0.8 * n)],
    "val": rest[int(0.8 * n) : int(0.9 * n)],
    "test": rest[int(0.9 * n) :],
    "test_unseen_templates": unseen,
}

# 4. Write each split to its own file
os.makedirs("data/splits", exist_ok=True)
for name, split in splits.items():
    with open(f"data/splits/{name}.jsonl", "w", encoding="utf-8") as f:
        for r in split:
            f.write(json.dumps(r) + "\n")
    print(name, len(split))

# 5. How many test sentences also appear in train (same words, different numbers)
train_sentences = {r["instruction"] for r in splits["train"]}
print("test sentences also in train:", sum(r["instruction"] in train_sentences for r in splits["test"]))