"""Generate the synthetic instruction dataset.

Version 2: four equally likely categories — translate only, rotate only,
scale only, and combined (2 or 3 edits) — with vague or explicit amounts.
"""
import json
import math
import random

from dyooptix.data import vocab
from dyooptix.data.distributions import EXPLICIT_RANGES, sample_size

DO_NOTHING = [0.0, 0.0, 1.0, 0.0, 0.0]   # [tx, ty, cos θ, sin θ, log s]
KIND_INDEX = {"translate": 0, "rotate": 1, "scale": 2}   # position in the presence list
REFERENCE_WIDTH = 512                    # explicit pixel amounts assume this width
EXPLICIT_SHARE = 0.25                    # fraction of edits with an exact number


def pick_amount(kind, rng):
    """How much? Returns (size, phrase, level). Size is positive, in model units."""
    if rng.random() < EXPLICIT_SHARE:
        low, high = EXPLICIT_RANGES[kind]
        if kind == "translate":
            pixels = rng.randint(low, high)
            return pixels / REFERENCE_WIDTH, f"by {pixels} pixels", "explicit"
        if kind == "rotate":
            degrees = rng.randint(low, high)
            return degrees, f"by {degrees} degrees", "explicit"
        factor = round(rng.uniform(low, high), 1)
        return math.log(factor), f"by a factor of {factor:g}", "explicit"

    level = rng.choice(list(vocab.ADVERBS))
    return sample_size(kind, level, rng), rng.choice(vocab.ADVERBS[level]), level


def add_edit(kind, size, rng, target):
    """Which way? Writes this edit's numbers into target; returns (verb, direction)."""
    if kind == "translate":
        direction = rng.choice(list(vocab.TRANSLATE_DIRECTIONS))
        dx, dy = vocab.TRANSLATE_DIRECTIONS[direction]
        target[0] = size * dx
        target[1] = size * dy
        return rng.choice(vocab.TRANSLATE_VERBS), direction

    if kind == "rotate":
        sign = rng.choice([-1, 1])
        theta = math.radians(size * sign)
        target[2] = math.cos(theta)
        target[3] = math.sin(theta)
        return rng.choice(vocab.ROTATE_VERBS), rng.choice(vocab.ROTATE_DIRECTIONS[sign])

    # scale: the verb decides bigger or smaller
    if rng.random() < 0.5:
        target[4] = size
        return rng.choice(vocab.SCALE_UP_VERBS), ""
    target[4] = -size
    return rng.choice(vocab.SCALE_DOWN_VERBS), ""


def make_example(rng):
    """Build one instruction (one, two or three edits) and its exact target."""
    category = rng.choice(["translate", "rotate", "scale", "combined"])
    if category == "combined":
        kinds = rng.sample(["translate", "rotate", "scale"], rng.choice([2, 3]))
    else:
        kinds = [category]

    target = DO_NOTHING.copy()
    presence = [0, 0, 0]
    clauses = []
    levels = []
    template_id = -1          # -1 for combined: built from clause patterns instead

    for i, kind in enumerate(kinds):
        size, amount, level = pick_amount(kind, rng)
        verb, direction = add_edit(kind, size, rng, target)
        presence[KIND_INDEX[kind]] = 1
        levels.append(level)

        obj = rng.choice(vocab.OBJECTS) if i == 0 else "it"
        if category == "combined":
            pattern = rng.choice(vocab.CLAUSE_TEMPLATES[kind])
        else:
            template_id = rng.randrange(len(vocab.TEMPLATES[kind]))
            pattern = vocab.TEMPLATES[kind][template_id]
        clauses.append(pattern.format(verb=verb, obj=obj, adv=amount, dir=direction))

    return {
        "instruction": rng.choice(vocab.JOINERS).join(clauses),
        "target": [round(v, 4) for v in target],
        "presence": presence,
        "category": category,
        "edits": kinds,
        "levels": levels,
        "template_id": template_id,
    }


def generate(n, seed, path):
    """Write n examples to a JSONL file, one example per line."""
    rng = random.Random(seed)
    with open(path, "w", encoding="utf-8") as f:
        for _ in range(n):
            f.write(json.dumps(make_example(rng)) + "\n")


if __name__ == "__main__":
    generate(n=5000, seed=42, path="data/dataset_v2.jsonl")
    print("wrote 5000 examples to data/dataset_v2.jsonl")