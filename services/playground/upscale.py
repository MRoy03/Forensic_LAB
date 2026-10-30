"""Resolution-enhancement placeholder.

Explicit placeholder: plain cubic-interpolation resize, standing in for a
future real super-resolution model (e.g. ESRGAN). Swap this implementation
once such a model is trained/available; the controller.py dispatch interface
stays the same.
"""

from __future__ import annotations

import cv2
import numpy as np


def apply_upscale(image_bgr: np.ndarray, scale: int = 2) -> tuple[np.ndarray, None]:
    upscaled = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return upscaled, None
