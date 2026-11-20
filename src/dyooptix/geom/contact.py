"""Translations derived from the scene: move one object until it touches another, or place it beside another."""
import numpy as np
from dyooptix.models.numbers import snap


def pixels(mask, dy=0.0):
    """(x, y) of the mask's pixels, optionally shifted down by dy."""
    ys, xs = np.nonzero(mask > 127)
    return xs.astype(float), ys.astype(float) + dy


def approach(mover, target, d, gap=0.0):
    """Shift along unit direction d that stops the mover `gap` px before the target. None if the target is not in its path.
    mover, target: (x, y) pixel arrays. Works lane by lane: lanes are 1 px wide, at right angles to d."""
    d = np.asarray(d, float) / np.hypot(*d)
    n = np.array([-d[1], d[0]])                                    # across the direction of motion
    mu, mv = mover[0] * d[0] + mover[1] * d[1], np.round(mover[0] * n[0] + mover[1] * n[1]).astype(int)
    tu, tv = target[0] * d[0] + target[1] * d[1], np.round(target[0] * n[0] + target[1] * n[1]).astype(int)

    lo = min(mv.min(), tv.min())
    lanes = max(mv.max(), tv.max()) - lo + 1
    front = np.full(lanes, -np.inf)
    np.maximum.at(front, mv - lo, mu)                              # mover's leading pixel in each lane
    back = np.full(lanes, np.inf)
    np.minimum.at(back, tv - lo, tu)                               # target's nearest pixel in each lane
    gaps = back - front
    shared = np.isfinite(gaps)                                     # lanes that hold both objects
    if not shared.any():
        return None
    t = gaps[shared].min() - 1 - gap                               # pixel centres 1 apart = touching
    return t * d


def touch(mover_mask, target_mask, d=None):
    """Move until touching. d: direction from the text. With no direction the mover slides along the main axis
    (horizontal or vertical, whichever centre distance is larger) so it stays on its surface; if the two share no
    lane on that axis it heads straight for the target's centre.
    Returns (tx, ty) in px, or None if the target is not in that direction."""
    mover, target = pixels(mover_mask), pixels(target_mask)
    if d is not None:
        shift = approach(mover, target, d)
        return None if shift is None or shift @ np.asarray(d, float) < 0 else tuple(shift)

    dx, dy = target[0].mean() - mover[0].mean(), target[1].mean() - mover[1].mean()
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return None                                                # same centre: no direction to go in
    axis = (np.sign(dx), 0.0) if abs(dx) >= abs(dy) else (0.0, np.sign(dy))
    for direction in (axis, (dx, dy)):                             # main axis first, then straight at the centre
        shift = approach(mover, target, direction)
        if shift is not None and shift @ np.asarray(direction, float) >= 0:
            return tuple(shift)
    return None


def beside(mover_mask, target_mask, side=None, gap_share=0.1):
    """Place the mover beside the target, bottoms aligned, with a gap of gap_share x the narrower width.
    side: "left" or "right" of the target; None picks the side the mover is already on. Returns (tx, ty) in px."""
    mover, target = pixels(mover_mask), pixels(target_mask)
    dy = target[1].max() - mover[1].max()                          # bottoms aligned
    if side is None:
        side = "left" if mover[0].mean() < target[0].mean() else "right"
    d = (1, 0) if side == "left" else (-1, 0)                      # approach the target from that side
    widths = np.ptp(mover[0]) + 1, np.ptp(target[0]) + 1          # in pixels
    gap = np.floor(gap_share * min(widths))
    shifted = (mover[0], mover[1] + dy)
    tx, _ = approach(shifted, target, d, gap)                      # type: ignore # bottoms aligned, so they always share a lane
    return float(tx), float(dy)


def to_edge(mover_mask, side=None, d=None):
    """Move until the mover touches the photo's border: the named side, else along d, else the nearest side.
    Returns (tx, ty) in px."""
    H, W = mover_mask.shape
    xs, ys = pixels(mover_mask)
    room = {"left": xs.min(), "right": W - 1 - xs.max(), "top": ys.min(), "bottom": H - 1 - ys.max()}
    unit = {"left": (-1, 0), "right": (1, 0), "top": (0, -1), "bottom": (0, 1)}
    if side is None and d is None:
        side = min(room, key=room.get)                             # nearest edge
    if side is not None:
        return tuple(room[side] * np.array(unit[side], float))
    d = np.asarray(d, float) / np.hypot(*d)
    steps = [room[s] / (np.array(u) @ d) for s, u in unit.items() if np.array(u) @ d > 1e-9]
    return tuple(min(steps) * d)                                   # the first border it reaches along d


def relation_shift(rel, mover_mask, target_mask, pred=None):
    """Translation (tx, ty) in px for a relation from relations.parse (touch, edge or beside).
    pred: the text model's [tx, ty, ...] for rel["rest"]; only its direction is used, for touch with a direction word.
    Raises ValueError with a readable reason when the move is not possible."""
    if rel["kind"] == "beside":
        return beside(mover_mask, target_mask, rel["side"])
    d = None                                                       # no direction word: slide toward the target
    if rel["kind"] == "edge" and (rel["side"] or not rel["has_direction"]):
        return to_edge(mover_mask, rel["side"])                    # named side, or the nearest one
    if rel["has_direction"]:
        if pred is None or np.hypot(pred[0], pred[1]) == 0:
            raise ValueError(f"no movement direction found in '{rel['rest']}'")
        d = snap(pred[0], pred[1], rel["rest"])                    # diagonal only if the text names one
    if rel["kind"] == "edge":
        return to_edge(mover_mask, d=d)
    shift = touch(mover_mask, target_mask, d)
    if shift is None:
        raise ValueError(f"the {rel['target']} is not in the {rel['mover'] or 'object'}'s path in that direction")
    return shift