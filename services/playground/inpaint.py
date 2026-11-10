"""Inpainting forgery generator: masks out a region and fills it via
cv2.inpaint, simulating content-aware object removal."""

from __future__ import annotations

import random

import cv2
import numpy as np


def apply_inpaint(
    image_bgr: np.ndarray, patch_size: tuple[int, int] = (64, 64)
) -> tuple[np.ndarray, np.ndarray]:
    h, w = image_bgr.shape[:2]
    pw, ph = patch_size
    pw = min(pw, w // 2) or 1
    ph = min(ph, h // 2) or 1

    x = random.randint(0, w - pw)
    y = random.randint(0, h - ph)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y : y + ph, x : x + pw] = 255

    inpainted = cv2.inpaint(image_bgr, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    return inpainted, mask
