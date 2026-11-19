"""Translations derived from the scene: move one object until it touches another, or place it beside another."""
import numpy as np


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
    """Move until touching. d: direction from the text, or None to head for the target's centre.
    Returns (tx, ty) in px, or None if the target is not in that direction."""
    mover, target = pixels(mover_mask), pixels(target_mask)
    if d is None:
        d = (target[0].mean() - mover[0].mean(), target[1].mean() - mover[1].mean())
    shift = approach(mover, target, d)
    return None if shift is None or shift @ np.asarray(d, float) < 0 else tuple(shift)


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
    tx, _ = approach(shifted, target, d, gap)                      # bottoms aligned, so they always share a lane
    return float(tx), float(dy)