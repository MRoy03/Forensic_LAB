"""Splicing forgery generator: Poisson-blends a patch into a different region,
optionally from a donor image (self-donor if none given)."""

from __future__ import annotations

import random

import cv2
import numpy as np


def apply_splice(
    image_bgr: np.ndarray,
    donor_bgr: np.ndarray | None = None,
    patch_size: tuple[int, int] = (80, 80),
) -> tuple[np.ndarray, np.ndarray]:
    donor = donor_bgr if donor_bgr is not None else image_bgr
    h, w = image_bgr.shape[:2]
    dh, dw = donor.shape[:2]
    pw, ph = patch_size

    pw = min(pw, w // 2, dw) or 1
    ph = min(ph, h // 2, dh) or 1

    src_x = random.randint(0, dw - pw)
    src_y = random.randint(0, dh - ph)
    patch = donor[src_y : src_y + ph, src_x : src_x + pw].copy()

    dst_x = random.randint(pw // 2 + 1, w - pw // 2 - 1)
    dst_y = random.randint(ph // 2 + 1, h - ph // 2 - 1)
    center = (dst_x, dst_y)

    patch_mask = np.full((ph, pw), 255, dtype=np.uint8)

    try:
        blended = cv2.seamlessClone(patch, image_bgr, patch_mask, center, cv2.NORMAL_CLONE)
    except cv2.error:
        # seamlessClone can fail on degenerate/edge-adjacent regions; fall back
        # to a hard paste so the playground still produces a result.
        blended = image_bgr.copy()
        blended[dst_y - ph // 2 : dst_y - ph // 2 + ph, dst_x - pw // 2 : dst_x - pw // 2 + pw] = patch

    mask = np.zeros((h, w), dtype=np.uint8)
    y0, x0 = dst_y - ph // 2, dst_x - pw // 2
    mask[y0 : y0 + ph, x0 : x0 + pw] = 255

    return blended, mask
