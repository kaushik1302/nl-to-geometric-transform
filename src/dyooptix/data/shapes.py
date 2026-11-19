"""Shape test set for scene-derived edits: a red circle, a blue square, a sentence with no number,
and the exact expected translation from circle-rectangle geometry (independent of geom/contact.py)."""
import csv
import math
import random
from pathlib import Path

import cv2
import numpy as np

SIZE = 512
OUT = Path("data/shapes")
AXES = {"right": (1, 0), "left": (-1, 0), "down": (0, 1), "up": (0, -1)}
TOUCH_VERBS = ["move", "push", "shift", "slide"]
TOUCH_WORDS = ["touches", "reaches", "hits"]
BESIDE_WORDS = ["beside", "next to", "near", "alongside", "close to"]


def along(direction, circle, square):
    """Circle centre and the square, in coordinates where the motion is +a and b runs across it."""
    (cx, cy, r), (x0, y0, x1, y1) = circle, square
    if direction in ("right", "left"):
        s = 1 if direction == "right" else -1
        return s * cx, cy, (x0 if s > 0 else -x1), (y0, y1)
    s = 1 if direction == "down" else -1
    return s * cy, cx, (y0 if s > 0 else -y1), (x0, x1)


def expected_touch(direction, circle, square):
    """Distance the circle moves until its pixels are next to the square's; None if it would pass by."""
    r = circle[2]
    a, b, face, (b0, b1) = along(direction, circle, square)
    off = 0 if b0 <= b <= b1 else min(abs(b - b0), abs(b - b1))   # how far the centre line misses the square
    if off > r:
        return None
    reach = math.sqrt(r * r - off * off)                          # circle's extent towards the square in the meeting lane
    return face - 1 - a - reach


def expected_beside(side, circle, square, gap_share=0.1):
    (cx, cy, r), (x0, y0, x1, y1) = circle, square
    ty = y1 - (cy + r)                                            # bottoms aligned
    off = max(0, y0 - (cy + ty))                                  # circle's widest row above the square's top?
    reach = math.sqrt(r * r - off * off)
    gap = math.floor(gap_share * min(2 * r + 1, x1 - x0 + 1))
    tx = x0 - 1 - gap - (cx + reach) if side == "left" else x1 + 1 + gap - (cx - reach)
    return tx, ty


def inside(circle, square, tx=0, ty=0):
    (cx, cy, r), (x0, y0, x1, y1) = circle, square
    return (r <= cx + tx < SIZE - r and r <= cy + ty < SIZE - r
            and 0 <= x0 and x1 < SIZE and 0 <= y0 and y1 < SIZE)


def apart(circle, square, margin=10):
    (cx, cy, r), (x0, y0, x1, y1) = circle, square
    return cx + r + margin < x0 or cx - r - margin > x1 or cy + r + margin < y0 or cy - r - margin > y1


def touch_case(rng, direction, named, corner):
    """Square ahead of the circle along `direction`; with corner, the circle's centre line misses the square's side."""
    while True:
        r, side = rng.randint(25, 55), rng.randint(50, 110)
        gap = rng.randint(20, 250)
        cx, cy = rng.randint(r, SIZE - r - 1), rng.randint(r, SIZE - r - 1)
        dx, dy = AXES[direction]
        offset = 0 if not named else rng.randint(-side // 3, side // 3)
        if corner:                                               # centre line passes beside the square, still within r
            offset = rng.choice([-1, 1]) * (side // 2 + rng.randint(5, r - 5))
        near = (cx + dx * (r + gap), cy + dy * (r + gap))          # point on the square's near side
        if dx:
            x0 = near[0] if dx > 0 else near[0] - side + 1
            y0 = cy + offset - side // 2
        else:
            y0 = near[1] if dy > 0 else near[1] - side + 1
            x0 = cx + offset - side // 2
        circle, square = (cx, cy, r), (x0, y0, x0 + side - 1, y0 + side - 1)
        t = expected_touch(direction, circle, square)
        if t is not None and t > 5 and inside(circle, square, dx * t, dy * t) and apart(circle, square):
            return circle, square, (dx * t, dy * t)


def beside_case(rng, side):
    """side: where the circle must end ("left"/"right" of the square), or None for the nearest side."""
    while True:
        r = rng.randint(25, 55)
        w = h = rng.randint(50, 120)                               # a real square, so "square" names it correctly
        circle = (rng.randint(r, SIZE - r - 1), rng.randint(r, SIZE - r - 1), r)
        x0, y0 = rng.randint(0, SIZE - w), rng.randint(0, SIZE - h)
        square = (x0, y0, x0 + w - 1, y0 + h - 1)
        end = side or ("left" if circle[0] < (square[0] + square[2]) / 2 else "right")
        tx, ty = expected_beside(end, circle, square)
        if apart(circle, square) and inside(circle, square, tx, ty) and abs(tx) > 20:
            return circle, square, (tx, ty), end


def edge_case(rng, mode):
    """mode: a side name ("left"...), "direction" (a direction word, no side) or "nearest" (neither)."""
    while True:
        r, w = rng.randint(25, 55), rng.randint(50, 110)
        circle = (rng.randint(r, SIZE - r - 1), rng.randint(r, SIZE - r - 1), r)
        x0, y0 = rng.randint(0, SIZE - w), rng.randint(0, SIZE - w)
        square = (x0, y0, x0 + w - 1, y0 + w - 1)
        cx, cy, r = circle
        room = {"left": cx - r, "right": SIZE - 1 - (cx + r), "up": cy - r, "down": SIZE - 1 - (cy + r)}
        ranked = sorted(room, key=room.get)
        if mode == "nearest" and room[ranked[1]] - room[ranked[0]] < 10:
            continue                                               # nearest edge must be clear-cut
        way = ranked[0] if mode == "nearest" else (rng.choice(list(AXES)) if mode == "direction" else mode)
        dx, dy = AXES[way]
        if apart(circle, square) and room[way] > 20:
            return circle, square, (dx * room[way], dy * room[way]), way


def draw(circle, square):
    image = np.full((SIZE, SIZE, 3), 255, np.uint8)
    cx, cy, r = circle
    x0, y0, x1, y1 = square
    cv2.rectangle(image, (x0, y0), (x1, y1), (50, 90, 200), -1)
    cv2.circle(image, (cx, cy), r, (220, 50, 50), -1)
    circle_mask = np.zeros((SIZE, SIZE), np.uint8)
    cv2.circle(circle_mask, (cx, cy), r, 255, -1)
    square_mask = np.zeros((SIZE, SIZE), np.uint8)
    square_mask[y0:y1 + 1, x0:x1 + 1] = 255
    return image, circle_mask, square_mask


def build(seed=643):
    rng = random.Random(seed)
    rows = []
    plan = ([("touch_dir", d, i % 3 == 2) for d in AXES for i in range(4)]        # 16, one per direction meets a corner
            + [("touch_nodir", d, False) for d in AXES for _ in range(2)]        # 8
            + [("beside", None, False)] * 16                                     # 16, nearest side
            + [("side", s, False) for s in ("left", "right") for _ in range(4)]  # 8, named side
            + [("edge", m, False) for m in ("left", "right", "up", "down", "direction", "direction",
                                            "nearest", "nearest")])               # 8, the photo's border
    for kind, arg, corner in plan:
        if kind.startswith("touch"):
            circle, square, (tx, ty) = touch_case(rng, arg, kind == "touch_dir", corner)
            word, verb = rng.choice(TOUCH_WORDS), rng.choice(TOUCH_VERBS)
            sentence = (f"{verb} the circle {arg} until it {word} the square" if kind == "touch_dir"
                        else f"{verb} the circle until it {word} the square")
            note = "corner" if corner else arg
        elif kind == "beside":
            circle, square, (tx, ty), end = beside_case(rng, None)
            sentence = f"{rng.choice(['put', 'move', 'place'])} the circle {rng.choice(BESIDE_WORDS)} the square"
            note = end
        elif kind == "edge":
            circle, square, (tx, ty), way = edge_case(rng, arg)
            edge = rng.choice(["edge", "end", "border"])
            named = {"up": "top", "down": "bottom"}.get(way, way) # type: ignore
            verb, word = rng.choice(TOUCH_VERBS), rng.choice(TOUCH_WORDS)
            if arg == "direction":
                sentence = f"{verb} the circle {way} until it {word} the {edge}"
            elif arg == "nearest":
                sentence = f"{verb} the circle until it {word} the {edge}"
            else:
                sentence = f"{verb} the circle until it {word} the {named} {edge}"
            note = f"{arg}:{way}"
        else:
            circle, square, (tx, ty), end = beside_case(rng, arg)
            sentence = f"{rng.choice(['move', 'place'])} the circle to the {arg} of the square"
            note = arg
        rows.append({"id": f"s{len(rows):02d}", "kind": kind, "note": note, "instruction": sentence,
                     "expected_tx": round(tx, 2), "expected_ty": round(ty, 2),
                     "circle": " ".join(map(str, circle)), "square": " ".join(map(str, square))})

    for sub in ("images", "masks"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    for row in rows:
        image, circle_mask, square_mask = draw(tuple(map(int, row["circle"].split())), tuple(map(int, row["square"].split())))
        cv2.imwrite(str(OUT / "images" / f"{row['id']}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(OUT / "masks" / f"{row['id']}_circle.png"), circle_mask)
        cv2.imwrite(str(OUT / "masks" / f"{row['id']}_square.png"), square_mask)
    with open(OUT / "shapes.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} images to {OUT}")


if __name__ == "__main__":
    build()