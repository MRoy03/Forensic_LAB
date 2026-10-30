"""Shared image -> model-input pipeline.

Used by the live app (feeding onnxruntime.InferenceSession.run directly),
and by the training notebooks (which wrap this same numpy output with
torch.from_numpy). Deliberately torch-free so it works identically in both
places without pulling PyTorch into the served app.
"""

from __future__ import annotations

import cv2
import numpy as np

from services.preprocessing.ela import compute_ela
from services.preprocessing.fft import compute_fft_magnitude


def decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode raw uploaded bytes into a BGR uint8 image array."""
    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image bytes (unsupported or corrupt format)")
    return image


def preprocess_image(image_bgr: np.ndarray, target_size: tuple[int, int] = (256, 256)) -> np.ndarray:
    """Resize -> normalize -> compute ELA + FFT -> stack into a (5, H, W)
    float32 array: [R, G, B, ELA, FFT], each channel scaled to [0, 1]."""
    resized_bgr = cv2.resize(image_bgr, target_size, interpolation=cv2.INTER_AREA)

    rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    ela = compute_ela(resized_bgr).astype(np.float32) / 255.0
    fft_mag = compute_fft_magnitude(resized_bgr).astype(np.float32) / 255.0

    stacked = np.concatenate(
        [
            np.transpose(rgb, (2, 0, 1)),  # (3, H, W)
            ela[np.newaxis, :, :],  # (1, H, W)
            fft_mag[np.newaxis, :, :],  # (1, H, W)
        ],
        axis=0,
    )
    return stacked.astype(np.float32)


def preprocess_image_batch(image_bgr: np.ndarray, target_size: tuple[int, int] = (256, 256)) -> np.ndarray:
    """Same as preprocess_image but with a leading batch dimension, ready for
    onnxruntime.InferenceSession.run(..., {"input": array})."""
    return preprocess_image(image_bgr, target_size)[np.newaxis, ...]
