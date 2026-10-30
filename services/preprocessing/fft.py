"""FFT magnitude spectrum: surfaces frequency-domain artifacts left by
GAN generation, resampling, and other synthetic manipulation pipelines."""

from __future__ import annotations

import cv2
import numpy as np


def compute_fft_magnitude(image_bgr: np.ndarray) -> np.ndarray:
    """Grayscale -> 2D FFT -> shifted log-magnitude spectrum, normalized to
    uint8 [0, 255]. Returns a single-channel array, same (H, W) as input."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1.0)

    normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)
