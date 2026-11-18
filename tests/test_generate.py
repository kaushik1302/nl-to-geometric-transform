import random
import re

from dyooptix.data.generate import make_example


def single_edit_examples(kind, n=4000):
    """Generate examples and keep those with exactly one edit of this kind."""
    rng = random.Random(0)
    return [e for e in (make_example(rng) for _ in range(n)) if e["edits"] == [kind]]


def test_rotate_left_right_signs():
    # left = counter-clockwise (sin > 0), right = clockwise (sin < 0)
    left = [e for e in single_edit_examples("rotate") if re.search(r"\bleft\b", e["instruction"])]
    right = [e for e in single_edit_examples("rotate") if re.search(r"\bright\b", e["instruction"])]
    assert left and right
    assert all(e["target"][3] > 0 for e in left)
    assert all(e["target"][3] < 0 for e in right)


def test_translate_left_right_signs():
    ex = single_edit_examples("translate")
    left = [e for e in ex if re.search(r"\bleft(wards)?\b", e["instruction"])]
    right = [e for e in ex if re.search(r"\bright(wards)?\b", e["instruction"])]
    assert left and right
    assert all(e["target"][0] < 0 for e in left)
    assert all(e["target"][0] > 0 for e in right)


def test_translate_percent_is_fraction_of_width():
    # "by 31%" on a straight move: |tx| + |ty| = 0.31
    ex = [e for e in single_edit_examples("translate")
          if re.search(r"\d+\s*(%|percent)", e["instruction"]) and "toward" not in e["instruction"]]
    assert ex
    for e in ex:
        n = int(re.search(r"(\d+)\s*(%|percent)", e["instruction"]).group(1)) # type: ignore
        assert abs(abs(e["target"][0]) + abs(e["target"][1]) - n / 100) < 1e-3