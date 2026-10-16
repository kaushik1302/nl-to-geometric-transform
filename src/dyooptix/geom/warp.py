import cv2
import numpy as np

from dyooptix.geom.affine import build_affine
from dyooptix.models.metrics import to_params


def warp(image, target, interpolation):
    """Apply the edit [tx, ty, cos, sin, log s] to the whole image, about its centre."""
    h, w = image.shape[:2]
    M = build_affine(to_params(target), (w / 2, h / 2), w)
    return cv2.warpAffine(image, M, (w, h), flags=interpolation)


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / union if union else 1.0