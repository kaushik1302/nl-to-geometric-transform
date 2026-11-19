import numpy as np

from dyooptix.geom.clamp import clamp, place

SIZE = (512, 512)


def box(x0, y0, x1, y1):
    points = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)
    return points, ((x0 + x1) / 2, (y0 + y1) / 2)


def inside(points, centre, tx, ty, deg, s):
    q = place(points, centre, SIZE, tx, ty, deg, s)
    return q.min() >= -1e-6 and q[:, 0].max() <= 511 + 1e-6 and q[:, 1].max() <= 511 + 1e-6


def test_translation_stops_at_right_edge():
    pts, c = box(300, 200, 420, 260)
    tx, ty, deg, s, warn = clamp(pts, c, SIZE, 200, 0, 0, 1)
    assert np.isclose(tx, 91) and "right edge" in warn[0]


def test_scale_grows_until_first_side_touches():
    pts, c = box(300, 200, 420, 260)
    tx, ty, deg, s, warn = clamp(pts, c, SIZE, 0, 0, 0, 3)
    assert np.isclose(s, 151 / 60) and "right edge" in warn[0]
    assert np.isclose(place(pts, c, SIZE, 0, 0, 0, s)[:, 0].max(), 511)


def test_scale_down_and_safe_edits_unchanged():
    pts, c = box(300, 200, 420, 260)
    assert clamp(pts, c, SIZE, 0, 0, 0, 0.5) == (0, 0, 0, 0.5, [])
    assert clamp(pts, c, SIZE, 50, -20, 10, 1.5) == (50, -20, 10, 1.5, [])


def test_rotation_near_edge_is_nudged_inward():
    pts, c = box(450, 200, 510, 260)
    tx, ty, deg, s, warn = clamp(pts, c, SIZE, 0, 0, 45, 1)
    assert deg == 45 and tx < 0 and inside(pts, c, tx, ty, deg, s)


def test_rotation_turned_back_when_object_too_big():
    pts, c = box(0, 0, 500, 500)
    tx, ty, deg, s, warn = clamp(pts, c, SIZE, 0, 0, 45, 1)
    assert 0 < deg < 45 and inside(pts, c, tx, ty, deg, s) and "rotation clamped" in warn[0]