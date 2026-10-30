"""Eigen-CAM: gradient-free class-activation-style explainability.

Kept at this path/name for continuity with the forgery-detection literature's
usual "Grad-CAM" terminology, but implemented as Eigen-CAM (Muhammad & Yeasin,
2020) rather than true gradient-based Grad-CAM: it needs only the encoder's
forward-pass feature map (already produced by the same ONNX session call used
for classification), not a backward pass, so it works with ONNX Runtime
serving and never requires PyTorch autograd at inference time.

If this project is ever run on hardware with PyTorch available at serving
time, this is the file to swap for pytorch_grad_cam.GradCAM / ScoreCAM for a
class-discriminative heatmap instead of this class-agnostic saliency map.
"""

from __future__ import annotations

import cv2
import numpy as np


def generate_eigencam(feature_map: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    """feature_map: shape (1, C, h, w) or (C, h, w) -- the model's last
    encoder feature map for a single sample. output_size: (width, height) to
    resize the heatmap to (typically the original image size).

    Returns a float32 heatmap normalized to [0, 1], shape (height, width).
    """
    arr = np.asarray(feature_map, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[0]  # drop batch dim -> (C, h, w)

    c, h, w = arr.shape
    flattened = arr.reshape(c, h * w)  # (C, h*w)

    # First principal component across channels via SVD on the (mean-centered)
    # channel matrix -- the dominant mode of variation across channels is
    # used as a class-agnostic saliency signal (Eigen-CAM).
    centered = flattened - flattened.mean(axis=1, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        projection = vt[0]  # (h*w,) -- top principal component in spatial domain
    except np.linalg.LinAlgError:
        projection = flattened.mean(axis=0)

    heatmap = projection.reshape(h, w)

    # Sign is arbitrary from SVD; orient so higher activation -> higher value.
    if np.mean(heatmap) < 0:
        heatmap = -heatmap

    heatmap -= heatmap.min()
    max_val = heatmap.max()
    if max_val > 1e-8:
        heatmap /= max_val

    width, height = output_size
    resized = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.float32)
