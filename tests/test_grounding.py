import numpy as np

from dyooptix.geom.grounding import box_iou, outline

def test_outline_of_a_rectangle_is_its_four_corners():
    mask = np.zeros((100, 100), np.uint8)
    mask[20:40, 30:70] = 255
    assert {tuple(p) for p in outline(mask)} == {(30, 20), (69, 20), (69, 39), (30, 39)}


def test_outline_ignores_a_small_stray_blob():
    mask = np.zeros((100, 100), np.uint8)
    mask[20:40, 30:70] = 255
    mask[90:93, 5:8] = 255
    assert outline(mask)[:, 0].min() == 30 and outline(mask)[:, 1].max() == 39

def test_box_iou():
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1
    assert box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0
    assert np.isclose(box_iou([0, 0, 10, 10], [5, 0, 15, 10]), 50 / 150)