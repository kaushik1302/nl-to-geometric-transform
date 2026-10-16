"""Generate the synthetic instruction dataset.

Version 3: four equally likely categories — translate only, rotate only,
scale only, and combined (2 or 3 edits) — with vague or explicit amounts.
v3 adds wider explicit ranges, several number formats (px, °, 2x), and
neutral scale verbs (scale/resize) whose factor gives the direction.
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
        if kind == "scale":
            return None, None, "explicit"     # decided in explicit_scale(): it needs the direction first
        low, high = EXPLICIT_RANGES[kind]
        phrase = rng.choice(vocab.EXPLICIT_PHRASES[kind])
        if kind == "translate":
            pixels = rng.randint(low, high)
            return pixels / REFERENCE_WIDTH, phrase.format(n=pixels), "explicit"
        degrees = rng.randint(low, high)
        return degrees, phrase.format(n=degrees), "explicit"

    level = rng.choice(list(vocab.ADVERBS))
    return sample_size(kind, level, rng), rng.choice(vocab.ADVERBS[level]), level


def explicit_scale(rng, target):
    """Explicit scale in one of three forms. Writes target[4]; returns (verb, phrase)."""
    low, high = EXPLICIT_RANGES["scale"]
    form = rng.choice(["bigger", "smaller_result", "smaller_divisor"])
    if form == "bigger":                      # "enlarge it by 2.5x", "scale it to 3x"
        n = round(rng.uniform(low, high), 1)
        target[4] = math.log(n)
        verbs = vocab.SCALE_UP_VERBS + vocab.SCALE_NEUTRAL_VERBS
        phrase = rng.choice(["by a factor of {n}", "by {n}x", "to {n}x"])
    elif form == "smaller_result":            # "shrink it to 0.7x", "resize it by 0.75x"
        n = round(rng.uniform(1 / high, 1 / low), 2)
        target[4] = math.log(n)
        verbs = vocab.SCALE_DOWN_VERBS + vocab.SCALE_NEUTRAL_VERBS
        phrase = rng.choice(["to {n}x", "by {n}x"])
    else:                                     # "shrink it by a factor of 2": shrink verbs only
        n = round(rng.uniform(low, high), 1)
        target[4] = -math.log(n)
        verbs = vocab.SCALE_DOWN_VERBS
        phrase = rng.choice(["by a factor of {n}", "by {n}x"])
    return rng.choice(verbs), phrase.format(n=f"{n:g}")


def add_edit(kind, size, amount, level, rng, target):
    """Which way? Writes this edit's numbers into target; returns (verb, direction, amount)."""
    if kind == "translate":
        direction = rng.choice(list(vocab.TRANSLATE_DIRECTIONS))
        dx, dy = vocab.TRANSLATE_DIRECTIONS[direction]
        target[0] = size * dx
        target[1] = size * dy
        return rng.choice(vocab.TRANSLATE_VERBS), direction, amount

    if kind == "rotate":
        sign = rng.choice([-1, 1])
        theta = math.radians(size * sign)
        target[2] = math.cos(theta)
        target[3] = math.sin(theta)
        return rng.choice(vocab.ROTATE_VERBS), rng.choice(vocab.ROTATE_DIRECTIONS[sign]), amount

    # scale
    if level == "explicit":
        verb, amount = explicit_scale(rng, target)
        return verb, "", amount
    if rng.random() < 0.5:                    # vague: the verb decides bigger or smaller
        target[4] = size
        return rng.choice(vocab.SCALE_UP_VERBS), "", amount
    target[4] = -size
    return rng.choice(vocab.SCALE_DOWN_VERBS), "", amount


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
        verb, direction, amount = add_edit(kind, size, amount, level, rng, target)
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