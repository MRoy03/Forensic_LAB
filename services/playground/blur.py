"""Gaussian blur, applied to the whole image or a specific region."""

from __future__ import annotations

import cv2
import numpy as np


def apply_blur(
    image_bgr: np.ndarray, ksize: int = 15, region: tuple[int, int, int, int] | None = None
) -> tuple[np.ndarray, np.ndarray | None]:
    ksize = ksize if ksize % 2 == 1 else ksize + 1

    if region is None:
        blurred = cv2.GaussianBlur(image_bgr, (ksize, ksize), 0)
        return blurred, None

    x, y, w, h = region
    result = image_bgr.copy()
    roi = result[y : y + h, x : x + w]
    result[y : y + h, x : x + w] = cv2.GaussianBlur(roi, (ksize, ksize), 0)

    mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    mask[y : y + h, x : x + w] = 255
    return result, mask
