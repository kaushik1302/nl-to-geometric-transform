import numpy as np

from dyooptix.geom.contact import beside, touch


def rect(x0, y0, x1, y1, size=200):
    """Mask with a filled rectangle covering x0..x1, y0..y1 (inclusive)."""
    mask = np.zeros((size, size), np.uint8)
    mask[y0:y1 + 1, x0:x1 + 1] = 255
    return mask


def test_touch_right_stops_with_no_gap():
    tx, ty = touch(rect(10, 40, 29, 59), rect(60, 30, 79, 69), (1, 0))
    assert np.isclose(tx, 30) and np.isclose(ty, 0)                # right side 29 -> 59, next to the target at 60


def test_touch_in_the_wrong_direction_is_refused():
    assert touch(rect(10, 40, 29, 59), rect(60, 30, 79, 69), (-1, 0)) is None


def test_touch_target_out_of_the_path_is_refused():
    assert touch(rect(10, 10, 29, 29), rect(60, 100, 79, 120), (1, 0)) is None


def test_touch_without_direction_heads_for_the_target():
    tx, ty = touch(rect(10, 10, 19, 19), rect(50, 50, 59, 59))
    assert np.isclose(tx, ty) and 29 < tx < 31                     # diagonal, corners meet


def test_beside_nearest_side_aligns_bottoms_and_leaves_a_gap():
    tx, ty = beside(rect(10, 40, 29, 79), rect(100, 60, 139, 99))  # mover is on the left
    assert (tx, ty) == (68, 20)                                    # ends at x 78..97: 2 px gap (10% of 20 px)


def test_beside_named_side_goes_around():
    tx, ty = beside(rect(10, 40, 29, 79), rect(100, 60, 139, 99), side="right")
    assert (tx, ty) == (132, 20)                                   # ends at x 142..161, right of the target


def test_who_moves_matters():
    a, b = rect(10, 40, 29, 79), rect(100, 60, 139, 99)
    assert beside(a, b) != beside(b, a)                            # bottle beside mug is not mug beside bottle
    assert beside(b, a) == (-68, -20)