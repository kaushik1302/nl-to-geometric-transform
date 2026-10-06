"""Generate the synthetic instruction dataset.

Version 1: single transforms only (translate, rotate or scale),
vague magnitude words only.
"""
import json
import math
import random

from dyooptix.data import vocab
from dyooptix.data.distributions import sample_size

# Target vector [tx, ty, cos θ, sin θ, log s] for "change nothing"
DO_NOTHING = [0.0, 0.0, 1.0, 0.0, 0.0]


def make_example(rng):
    """Build one instruction and its exact target."""
    kind = rng.choice(["translate", "rotate", "scale"])
    level = rng.choice(list(vocab.ADVERBS))
    adverb = rng.choice(vocab.ADVERBS[level])
    size = sample_size(kind, level, rng)
    target = DO_NOTHING.copy()

    if kind == "translate":
        direction = rng.choice(list(vocab.TRANSLATE_DIRECTIONS))
        dx, dy = vocab.TRANSLATE_DIRECTIONS[direction]
        target[0] = size * dx
        target[1] = size * dy
        verb = rng.choice(vocab.TRANSLATE_VERBS)
        presence = [1, 0, 0]

    elif kind == "rotate":
        sign = rng.choice([-1, 1])
        direction = rng.choice(vocab.ROTATE_DIRECTIONS[sign])
        theta = math.radians(size * sign)
        target[2] = math.cos(theta)
        target[3] = math.sin(theta)
        verb = rng.choice(vocab.ROTATE_VERBS)
        presence = [0, 1, 0]

    else:  # scale: the verb decides bigger or smaller
        direction = ""
        if rng.random() < 0.5:
            verb = rng.choice(vocab.SCALE_UP_VERBS)
            target[4] = size
        else:
            verb = rng.choice(vocab.SCALE_DOWN_VERBS)
            target[4] = -size
        presence = [0, 0, 1]

    templates = vocab.TEMPLATES[kind]
    template_id = rng.randrange(len(templates))
    instruction = templates[template_id].format(
        verb=verb, obj=rng.choice(vocab.OBJECTS), adv=adverb, dir=direction)

    return {
        "instruction": instruction,
        "target": [round(v, 4) for v in target],
        "presence": presence,
        "edit_type": kind,
        "level": level,
        "adverb": adverb,
        "template_id": template_id,
    }


def generate(n, seed, path):
    """Write n examples to a JSONL file, one example per line."""
    rng = random.Random(seed)
    with open(path, "w", encoding="utf-8") as f:
        for _ in range(n):
            f.write(json.dumps(make_example(rng)) + "\n")


if __name__ == "__main__":
    generate(n=5000, seed=42, path="data/dataset_v1.jsonl")
    print("wrote 5000 examples to data/dataset_v1.jsonl")