import numpy as np

from dyooptix.geom.affine import build_affine

SIZE = 512  # reference image width and height in pixels
CENTRE = (SIZE / 2, SIZE / 2)
# The 4 image corners as columns [x, y, 1], so that M @ CORNERS moves all of them at once
CORNERS = np.array([[0, SIZE, 0, SIZE],
                    [0, 0, SIZE, SIZE],
                    [1, 1, 1, 1]], dtype=float)


def to_params(row):
    return {"tx": row[0], "ty": row[1], "cos": row[2], "sin": row[3], "log_s": row[4]}


def angle_deg(y):
    return np.degrees(np.arctan2(y[:, 3], y[:, 2]))


def corner_error(y_true, y_pred):
    errors = []
    for t, p in zip(y_true, y_pred):
        M_true = build_affine(to_params(t), CENTRE, SIZE)
        M_pred = build_affine(to_params(p), CENTRE, SIZE)
        moved = (M_true - M_pred) @ CORNERS          # 2 x 4: how far apart each corner lands
        errors.append(np.linalg.norm(moved, axis=0).mean())
    return np.mean(errors)


def evaluate(y_true, y_pred):
    angle_diff = angle_deg(y_pred) - angle_deg(y_true)
    angle_diff = (angle_diff + 180) % 360 - 180      # wrap into [-180, 180)
    return {
        "tx_px": np.abs(y_pred[:, 0] - y_true[:, 0]).mean() * SIZE,
        "ty_px": np.abs(y_pred[:, 1] - y_true[:, 1]).mean() * SIZE,
        "theta_deg": np.abs(angle_diff).mean(),
        "log_s": np.abs(y_pred[:, 4] - y_true[:, 4]).mean(),
        "corner_px": corner_error(y_true, y_pred),
    }