"""Copy-move forgery generator: duplicates a patch elsewhere in the same image."""

from __future__ import annotations

import random

import numpy as np


def apply_copy_move(
    image_bgr: np.ndarray, patch_size: tuple[int, int] = (64, 64), num_patches: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    result = image_bgr.copy()
    mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    h, w = image_bgr.shape[:2]
    pw, ph = patch_size

    pw = min(pw, w // 2) or 1
    ph = min(ph, h // 2) or 1

    for _ in range(num_patches):
        src_x = random.randint(0, w - pw)
        src_y = random.randint(0, h - ph)
        dst_x = random.randint(0, w - pw)
        dst_y = random.randint(0, h - ph)

        patch = image_bgr[src_y : src_y + ph, src_x : src_x + pw].copy()
        result[dst_y : dst_y + ph, dst_x : dst_x + pw] = patch
        mask[dst_y : dst_y + ph, dst_x : dst_x + pw] = 255

    return result, mask
