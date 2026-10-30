"""Error Level Analysis: highlights regions with inconsistent JPEG compression,
a classic forensic signal for splicing/compositing."""

from __future__ import annotations

import cv2
import numpy as np


def compute_ela(image_bgr: np.ndarray, quality: int = 90, scale: float = 15.0) -> np.ndarray:
    """Re-encode the image as JPEG at `quality`, diff against the original, and
    amplify by `scale`. Returns a single-channel uint8 array, same (H, W) as input."""
    ok, encoded = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("Failed to JPEG-encode image for ELA")

    recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    diff = cv2.absdiff(image_bgr, recompressed).astype(np.float32)
    diff_gray = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)

    amplified = np.clip(diff_gray * scale, 0, 255)
    return amplified.astype(np.uint8)
