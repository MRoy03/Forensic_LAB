"""Interpret raw mask_logits into a binary tampered-area mask at the original
image resolution."""

from __future__ import annotations

import cv2
import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def interpret_mask(
    mask_logits: np.ndarray, original_size: tuple[int, int], threshold: float
) -> np.ndarray:
    """mask_logits: shape (1, 1, H, W) or (1, H, W) or (H, W) for a single
    sample. original_size: (width, height) of the source image.
    Returns a uint8 array (0 or 255) resized to original_size."""
    arr = np.asarray(mask_logits)
    arr = arr.reshape(arr.shape[-2], arr.shape[-1])

    probs = _sigmoid(arr)
    binary = (probs > threshold).astype(np.uint8) * 255

    width, height = original_size
    return cv2.resize(binary, (width, height), interpolation=cv2.INTER_NEAREST)
