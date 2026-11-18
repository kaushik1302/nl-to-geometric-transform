"""Words and sentence patterns used by the dataset generator.

GeoBench's distinctive words (drag, swivel, contract, zoom, mildly, markedly,
appreciably, intensely, heavily, gently, lightly, strongly, "around the z-axis",
"uniformly") are deliberately NOT used here, so that GeoBench remains a test
of wording the model has never seen.
"""

# What the instruction refers to (generic nouns, since we use a synthetic canvas)
OBJECTS = ["the shape", "the object", "the square", "the circle", "the box", "it", "the image", "the picture"]

# Verbs, grouped by the transform they imply
TRANSLATE_VERBS = ["move", "shift", "nudge", "push"]
ROTATE_VERBS = ["rotate", "turn", "spin"]
SCALE_UP_VERBS = ["enlarge", "grow"]
SCALE_DOWN_VERBS = ["shrink", "reduce"]

# Translation direction -> unit vector (dx, dy) in image coordinates (y points DOWN)
D = 0.7071  # 1/sqrt(2), so diagonal moves are the same length as straight ones
# Translation direction (dx, dy) in image coordinates (y points DOWN) -> phrases for it.
# Grouped by direction so the generator picks each direction evenly,
# however many phrases each has (same idea as ROTATE_DIRECTIONS).
TRANSLATE_PHRASES = {
    (-1, 0):  ["to the left", "left", "leftwards"],
    (1, 0):   ["to the right", "right", "rightwards"],
    (0, -1):  ["up", "upwards"],
    (0, 1):   ["down", "downwards"],
    (-D, -D): ["toward the top-left"],
    (D, -D):  ["toward the top-right"],
    (-D, D):  ["toward the bottom-left"],
    (D, D):   ["toward the bottom-right"],
}
# Flat phrase -> vector lookup (used by encode.py for keyword flags)
TRANSLATE_DIRECTIONS = {p: v for v, phrases in TRANSLATE_PHRASES.items() for p in phrases}

# Rotation sign -> phrases for it (positive = counter-clockwise on screen).
# Grouped by sign so the generator can pick each direction 50/50,
# regardless of how many phrases each has.
ROTATE_DIRECTIONS = {
    -1: ["clockwise", "to the right", "right"],
    1:  ["counter-clockwise", "anticlockwise", "counterclockwise", "anti-clockwise",
         "to the left", "left"],
}

# Vague magnitude words, grouped by level. Three levels, matching GeoBench;
# each level's number range is in distributions.py.
ADVERBS = {
    "small": ["slightly", "a little", "a bit"],
    "medium": ["moderately", "somewhat"],
    "large": ["significantly", "a lot", "drastically"],
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

# Plain clause patterns for combined instructions ("... and ...").
# No "could you" prefix: it would sound wrong in the middle of a sentence.
CLAUSE_TEMPLATES = {
    "translate": ["{verb} {obj} {adv} {dir}", "{verb} {obj} {dir} {adv}"],
    "rotate": ["{verb} {obj} {adv} {dir}", "{verb} {obj} {dir} {adv}"],
    "scale": ["{verb} {obj} {adv}"],
}

# Words that join clauses in combined instructions
JOINERS = [" and ", ", then "]

# Sentence patterns held out of training: only in the held-out-template test set.
# These are the "can you ..." patterns, the only templates with that opening.
HELD_OUT_TEMPLATES = {"translate": 3, "rotate": 3, "scale": 2}

# v3: verbs that give no direction; the explicit factor does (2x bigger, 0.5x smaller)
SCALE_NEUTRAL_VERBS = ["scale", "resize"]

# v3: several ways of writing an explicit number; {n} is filled in by the generator
EXPLICIT_PHRASES = {
    "translate": ["by {n} pixels", "by {n} px", "{n} pixels", "{n} px"],
    "rotate": ["by {n} degrees", "by {n}°", "{n} degrees", "by {n} deg"],
}

# v4: percentages. Translate: % of image width (10% = 0.1 W).
# Scale: the verb gives the direction (enlarge by 20% = x1.2, reduce by 10% = x0.9).
PERCENT_PHRASES = {
    "translate": ["by {n}%", "{n}%", "by {n} percent"],
    "scale": ["by {n}%", "by {n} percent"],
}

# v4: "times" as another way to write an explicit scale factor (enlarge by 3 times = x3)
TIMES_PHRASES = ["by {n} times", "{n} times"]

# v4: whole scale clauses whose word gives the factor; {obj} is filled by the generator
FACTOR_CLAUSES = {
    2.0: ["double {obj}", "make {obj} twice as big", "make {obj} twice the size"],
    0.5: ["halve {obj}", "make {obj} half the size", "make {obj} half as big"],
}
FACTOR_WORDS = ["double", "twice", "halve", "half"]   # keyword flags for encode.py