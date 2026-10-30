"""Blend masks/heatmaps onto the original image for human-readable output."""

from __future__ import annotations

import cv2
import numpy as np


def overlay_mask(
    image_bgr: np.ndarray, mask: np.ndarray, color: tuple[int, int, int] = (0, 0, 255), alpha: float = 0.4
) -> np.ndarray:
    """mask: uint8 array (0 or 255), same (H, W) as image_bgr."""
    overlay = image_bgr.copy()
    colored = np.zeros_like(image_bgr)
    colored[:, :] = color
    mask_bool = mask > 0
    overlay[mask_bool] = cv2.addWeighted(image_bgr, 1 - alpha, colored, alpha, 0)[mask_bool]
    return overlay


def overlay_heatmap(image_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """heatmap: float32 array in [0, 1], same (H, W) as image_bgr."""
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0, 1))
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 1 - alpha, colored, alpha, 0)
