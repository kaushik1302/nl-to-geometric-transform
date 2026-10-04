import numpy as np
import cv2
from dyooptix.geom.affine import build_affine


def make_params(tx=0.0, ty=0.0, deg=0.0, s=1.0):
    """Turn readable values (degrees, plain scale) into the dict build_affine expects."""
    theta = np.radians(deg)
    return {"tx": tx, "ty": ty,
            "cos": np.cos(theta), "sin": np.sin(theta),
            "log_s": np.log(s)}


def apply(M, point):
    """Move one point (x, y) with a 2x3 matrix: append 1, then multiply."""
    x, y = point
    return M @ np.array([x, y, 1.0])


def test_identity():
    M = build_affine(make_params(), centroid=(37, 81), width=256)
    assert np.allclose(M, [[1, 0, 0], [0, 1, 0]])


def test_rotate_90_ccw_moves_point_up():
    # Centre (50, 50). A point 10 px right of centre, turned 90°
    # counter-clockwise on screen, should end 10 px ABOVE centre: (50, 40).
    M = build_affine(make_params(deg=90), centroid=(50, 50), width=100)
    assert np.allclose(apply(M, (60, 50)), [50, 40])


def test_centre_stays_fixed():
    c = (120, 45)
    M = build_affine(make_params(deg=-37, s=1.8), centroid=c, width=256)
    assert np.allclose(apply(M, c), c)


def test_translation_in_pixels():
    # Width 200: tx = 0.1 → 20 px right; ty = −0.05 → 10 px up.
    M = build_affine(make_params(tx=0.1, ty=-0.05), centroid=(64, 64), width=200)
    assert np.allclose(apply(M, (10, 10)), [30, 0])


def test_round_trip():
    M = build_affine(make_params(tx=0.07, ty=-0.12, deg=23, s=1.4), centroid=(90, 70), width=256)
    M_inv = cv2.invertAffineTransform(M)
    for p in [(0, 0), (255, 0), (128, 200)]:
        assert np.allclose(apply(M_inv, apply(M, p)), p)


def test_matches_opencv():
    cases = [(30, 1.5, (100, 80)), (-45, 0.5, (10, 200)),
             (170, 2.0, (128, 128)), (-179, 0.25, (0, 0))]
    for deg, s, c in cases:
        M = build_affine(make_params(deg=deg, s=s), centroid=c, width=256)
        assert np.allclose(M, cv2.getRotationMatrix2D(c, deg, s))