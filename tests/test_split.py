import numpy as np

from dyooptix.models.split import merge, split_clauses

NOTHING = [0, 0, 1, 0, 0]


def test_split_joiners():
    assert split_clauses("move it left and then rotate it, then shrink it; grow it") == \
        ["move it left", "rotate it", "shrink it", "grow it"]
    assert split_clauses("move it left, and enlarge it") == ["move it left", "enlarge it"]
    assert split_clauses("rotate it by 30 degrees") == ["rotate it by 30 degrees"]


def test_merge_adds_moves_angles_and_log_scales():
    a, b = np.radians(20), np.radians(-50)
    preds = [[0.1, 0, 1, 0, 0],                       # move right 0.1
             [0, -0.05, np.cos(a), np.sin(a), 0],     # move up 0.05, turn 20° ccw
             [0, 0, np.cos(b), np.sin(b), np.log(2)], # turn 50° cw, double
             NOTHING]                                 # a clause with no edit
    m = merge(preds)
    assert np.allclose(m[:2], [0.1, -0.05])
    assert np.isclose(np.degrees(np.arctan2(m[3], m[2])), -30)
    assert np.isclose(np.exp(m[4]), 2)


def test_merge_one_clause_is_unchanged():
    row = [0.2, -0.1, np.cos(0.3), np.sin(0.3), 0.4]
    assert np.allclose(merge([row]), row)

def test_diagonal_and_is_not_split():
    assert split_clauses("drag it diagonally up and right") == ["drag it diagonally up and right"]
    assert split_clauses("move it up and to the left by 40 px and rotate it right") == \
        ["move it up and to the left by 40 px", "rotate it right"]
    assert split_clauses("move it up and rotate it clockwise") == ["move it up", "rotate it clockwise"]