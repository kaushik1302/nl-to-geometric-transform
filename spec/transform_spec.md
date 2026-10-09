# Transform Specification — DyoOptix

## Coordinate system
- Origin at the image's top-left corner; x increases right, y increases DOWN.
- Pixels internally; the model works in normalised units (below).

## Transform family
2D similarity transform: translation, rotation, uniform scale.

## Model target vector
y = [t_x, t_y, cos θ, sin θ, log s]

- t_x, t_y : fraction of image width W, range [−0.5, 0.5].
             Positive t_x = right, positive t_y = DOWN ("move up" → t_y < 0).
- θ        : positive = counter-clockwise ON SCREEN
             (same convention as cv2.getRotationMatrix2D).
             Degrees for people, radians in code.
- s        : uniform scale in [0.25, 4]; stored as log s in [−1.386, 1.386].
- presence : p = [p_trans, p_rot, p_scale], 1 if the instruction mentions it.
- "do nothing" target: [0, 0, 1, 0, 0].

## Composition
M = T(t) · T(c) · R(θ) · S(s) · T(−c), applied right to left.
c = image centre = (W/2, H/2): edits apply to the whole image.

2×3 form for cv2.warpAffine, with α = s·cos θ, β = s·sin θ, t in pixels:
  [ α    β    t_x + (1−α)·c_x − β·c_y ]
  [ −β   α    t_y + β·c_x + (1−α)·c_y ]

## Interpolation
Images: bilinear. Masks: nearest-neighbour (never blend label values).

## External data: GeoBench 2D
- edit_param assumed [tx, ty, tz, rx, ry, rz, sx, sy, sz] — verify on load.
- θ = −rz   (GeoBench uses clockwise-positive).
- t = pixels / 512.
- s = sx   (sx = sy, uniform).

## Verification
Unit tests: identity, round trip, known corner, agreement with OpenCV.