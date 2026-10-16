"""Number ranges for each magnitude level, calibrated against GeoBench 2D.

Measured with tools/geobench_stats.py (min and max per level, rounded).
Units: translate = fraction of image width; rotate = degrees; scale = |log s|.
Sizes here are always positive: the sign (left/right, clockwise,
bigger/smaller) comes from the words in vocab.py.
"""

RANGES = {
    "translate": {"small": (0.05, 0.13), "medium": (0.10, 0.27), "large": (0.20, 0.47)},
    "rotate":    {"small": (5, 9),       "medium": (10, 19),     "large": (20, 39)},
    "scale":     {"small": (0.10, 0.26), "medium": (0.23, 0.51), "large": (0.41, 1.06)},
}

# Ranges for explicit numbers ("by 30 degrees"). Wider than GeoBench on purpose:
# an explicit number carries its own size, so only vague words need calibrating.
# translate: whole pixels on a 512-px reference width; rotate: whole degrees
# (180 covers every distinct rotation); scale: a factor, for enlarging and shrinking.
EXPLICIT_RANGES = {
    "translate": (20, 300),
    "rotate": (5, 180),
    "scale": (1.1, 4.0),
}


def sample_size(kind, level, rng):
    """Draw one size for this edit type and level, uniformly from its range.

    kind  : "translate", "rotate" or "scale"
    level : "small", "medium" or "large"
    rng   : a random generator (random.Random), created once with a fixed seed
    """
    low, high = RANGES[kind][level]
    return rng.uniform(low, high)