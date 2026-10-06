"""Words and sentence patterns used by the dataset generator.

GeoBench's distinctive words (drag, swivel, contract, zoom, mildly, markedly,
appreciably, intensely, heavily, gently, lightly, strongly, "around the z-axis",
"uniformly") are deliberately NOT used here, so that GeoBench remains a test
of wording the model has never seen.
"""

# What the instruction refers to (generic nouns, since we use a synthetic canvas)
OBJECTS = ["the shape", "the object", "the square", "the circle", "the box", "it"]

# Verbs, grouped by the transform they imply
TRANSLATE_VERBS = ["move", "shift", "nudge", "push", "slide"]
ROTATE_VERBS = ["rotate", "turn", "spin"]
SCALE_UP_VERBS = ["enlarge", "grow", "expand"]
SCALE_DOWN_VERBS = ["shrink", "reduce"]

# Translation direction -> unit vector (dx, dy) in image coordinates (y points DOWN)
D = 0.7071  # 1/sqrt(2), so diagonal moves are the same length as straight ones
TRANSLATE_DIRECTIONS = {
    "to the left": (-1, 0),
    "to the right": (1, 0),
    "up": (0, -1),
    "down": (0, 1),
    "toward the top-left": (-D, -D),
    "toward the top-right": (D, -D),
    "toward the bottom-left": (-D, D),
    "toward the bottom-right": (D, D),
}

# Rotation direction -> sign of θ (positive = counter-clockwise on screen)
ROTATE_DIRECTIONS = {
    "clockwise": -1,
    "counter-clockwise": 1,
    "anticlockwise": 1,
}

# Vague magnitude words, grouped by level. Three levels, matching GeoBench;
# each level's number range is in distributions.py.
ADVERBS = {
    "small": ["slightly", "a little", "a bit"],
    "medium": ["moderately", "somewhat"],
    "large": ["significantly", "a lot", "drastically"],
}

# Never used in training: only in the held-out-adverb test split (RQ2)
HELD_OUT_ADVERBS = {
    "a touch": "small",
    "considerably": "large",
}

# Sentence patterns; the generator fills {verb} {obj} {adv} {dir}.
# A template's position in its list is its id, so whole templates can be held out.
TEMPLATES = {
    "translate": [
        "{verb} {obj} {adv} {dir}",
        "{verb} {obj} {dir} {adv}",
        "could you {verb} {obj} {adv} {dir}",
        "can you {verb} {obj} {dir} {adv}",
    ],
    "rotate": [
        "{verb} {obj} {adv} {dir}",
        "{verb} {obj} {dir} {adv}",
        "could you {verb} {obj} {adv} {dir}",
        "can you {verb} {obj} {dir} {adv}",
    ],
    "scale": [
        "{verb} {obj} {adv}",
        "could you {verb} {obj} {adv}",
        "can you {verb} {obj} {adv}",
    ],
}