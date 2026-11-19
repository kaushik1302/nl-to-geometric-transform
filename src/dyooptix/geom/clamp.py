import numpy as np

from dyooptix.geom.affine import build_affine


def place(points, centre, size, tx, ty, deg, s):
    """Where the object's outline points land, with the same matrix as the image warp. tx, ty in pixels."""
    w = size[0]
    a = np.radians(deg)
    p = {"tx": tx / w, "ty": ty / w, "cos": np.cos(a), "sin": np.sin(a), "log_s": np.log(s)}
    M = build_affine(p, centre, w)
    return points @ M[:, :2].T + M[:, 2]


def slide(name, value, lo, hi, edges):
    """Clip one move component into [lo, hi]; returns the new value and a warning (or None)."""
    new = min(max(value, lo), hi)
    if abs(new - value) < 0.5:               # under half a pixel: not worth a warning
        return new, None
    edge = edges[0] if new == lo else edges[1]
    return new, f"⚠ {name} move clamped: {value:+.0f} px → {new:+.0f} px (object reached the {edge} edge)"


def clamp(points, centre, size, tx, ty, deg, s):
    """Keep the whole object inside the image. Order: scale, then rotation, then translation.
    points: (N, 2) outline in pixels (x, y); centre: (cx, cy); size: (W, H); tx, ty in pixels.
    Returns (tx, ty, deg, s, warnings)."""
    W, H = size[0] - 1, size[1] - 1          # last pixel index: the object must stay within 0..W, 0..H
    cx, cy = centre
    xmin, ymin = points.min(axis=0)
    xmax, ymax = points.max(axis=0)
    warnings = []

    # 1. scale: largest factor about the centre before the first side touches the border
    room = {"left": (cx, cx - xmin), "right": (W - cx, xmax - cx),
            "top": (cy, cy - ymin), "bottom": (H - cy, ymax - cy)}
    limits = {side: border / reach for side, (border, reach) in room.items() if reach > 0}
    side = min(limits, key=limits.get)
    if s > limits[side]:                     # scaling down never reaches this
        warnings.append(f"⚠ scale clamped: ×{s:.2f} → ×{limits[side]:.2f} (object reached the {side} edge)")
        s = limits[side]

    # 2. rotation: turn back toward 0° only if the turned object is too big for the image anywhere
    def fits(d):
        q = place(points, centre, size, 0, 0, d, s)
        return np.ptp(q[:, 0]) <= W + 1e-6 and np.ptp(q[:, 1]) <= H + 1e-6

    asked = deg
    while deg != 0 and not fits(deg):
        deg = np.sign(deg) * max(abs(deg) - 1, 0)
    if deg != asked:
        warnings.append(f"⚠ rotation clamped: {asked:+.0f}° → {deg:+.0f}° (object too large to turn fully)")

    # 3. translation: the range of moves that keeps the turned, scaled object inside
    q = place(points, centre, size, 0, 0, deg, s)
    tx, wx = slide("x", tx, -q[:, 0].min(), W - q[:, 0].max(), ("left", "right"))
    ty, wy = slide("y", ty, -q[:, 1].min(), H - q[:, 1].max(), ("top", "bottom"))
    warnings += [w for w in (wx, wy) if w]
    return tx, ty, deg, s, warnings