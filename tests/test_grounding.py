import numpy as np

from dyooptix.geom.grounding import outline


def test_outline_of_a_rectangle_is_its_four_corners():
    mask = np.zeros((100, 100), np.uint8)
    mask[20:40, 30:70] = 255
    assert {tuple(p) for p in outline(mask)} == {(30, 20), (69, 20), (69, 39), (30, 39)}


def test_outline_ignores_a_small_stray_blob():
    mask = np.zeros((100, 100), np.uint8)
    mask[20:40, 30:70] = 255
    mask[90:93, 5:8] = 255
    assert outline(mask)[:, 0].min() == 30 and outline(mask)[:, 1].max() == 39