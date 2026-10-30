"""Simulates aggressive JPEG re-compression artifacts."""

from __future__ import annotations

import cv2
import numpy as np


def apply_compression_artifact(image_bgr: np.ndarray, quality: int = 20) -> tuple[np.ndarray, None]:
    ok, encoded = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("Failed to JPEG-encode image for compression artifact")
    degraded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return degraded, None
