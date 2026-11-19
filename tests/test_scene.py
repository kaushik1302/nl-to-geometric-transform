import numpy as np

from dyooptix.geom.contact import beside
from dyooptix.models.scene import read, scene_edit

NONE = [0, 0, 1, 0, 0]


def rect(x0, y0, x1, y1, size=200):
    mask = np.zeros((size, size), np.uint8)
    mask[y0:y1 + 1, x0:x1 + 1] = 255
    return mask


def test_it_refers_to_the_first_named_object():
    clauses, rels, mover, texts = read("rotate the mug by 20 deg clockwise, scale it to 1.5x and move it beside the cat")
    assert clauses == ["rotate the mug by 20 deg clockwise", "scale it to 1.5x", "move it beside the cat"]
    assert mover == "mug" and rels[2]["kind"] == "beside" and rels[2]["target"] == "cat"
    assert texts[2] == "move it"                                   # only the part before "beside" goes to the model


def test_beside_is_measured_after_scaling():
    clauses, rels, mover, texts = read("enlarge the box 2x and put it next to the bin")
    mover_mask, target_mask = rect(40, 60, 59, 79), rect(120, 40, 179, 119)
    edit = scene_edit(clauses, rels, [[0, 0, 1, 0, np.log(2)], NONE], mover_mask, target_mask)
    assert np.isclose(np.exp(edit[4]), 2)                          # the 2x is kept
    assert abs(edit[0] * 200 - 46) <= 1 and abs(edit[1] * 200 - 30) <= 1   # as for the 40 x 40 doubled box
    assert edit[0] * 200 != beside(mover_mask, target_mask)[0]     # differs from measuring the unscaled box


def test_relation_replaces_a_plain_move():
    clauses, rels, mover, texts = read("move the box left and push it right until it touches the bin")
    edit = scene_edit(clauses, rels, [[-0.3, 0, 1, 0, 0], [0.2, 0, 1, 0, 0]], rect(10, 40, 29, 59), rect(60, 30, 79, 69))
    assert np.isclose(edit[0] * 200, 30) and np.isclose(edit[1], 0)


def test_px_in_a_plain_clause_means_pixels_of_this_photo():
    clauses, rels, mover, texts = read("move the box 50 px right")
    edit = scene_edit(clauses, rels, [[50 / 512, 0, 1, 0, 0]], rect(10, 40, 29, 59), None)
    assert np.isclose(edit[0] * 200, 50)