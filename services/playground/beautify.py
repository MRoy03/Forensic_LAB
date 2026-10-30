"""Simulated skin-smoothing / beautification filter (bilateral filter + tone tweak)."""

from __future__ import annotations

import cv2
import numpy as np


def apply_beautify(image_bgr: np.ndarray) -> tuple[np.ndarray, None]:
    smoothed = cv2.bilateralFilter(image_bgr, d=9, sigmaColor=75, sigmaSpace=75)
    brightened = cv2.convertScaleAbs(smoothed, alpha=1.05, beta=8)
    return brightened, None
