"""Encode numpy image arrays to base64 PNG strings for JSON responses."""

from __future__ import annotations

import base64

import cv2
import numpy as np


def array_to_base64_png(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Failed to PNG-encode array")
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def base64_png_to_array(data: str) -> np.ndarray:
    decoded = base64.b64decode(data)
    buffer = np.frombuffer(decoded, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode base64 PNG string")
    return image
