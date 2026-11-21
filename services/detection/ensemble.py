"""Pretrained "second opinion" image model (TensorFold/detectra-v1), loaded
and run independently of the custom 5-channel RGB+ELA+FFT model. Own
preprocessing path -- this model expects plain 3-channel RGB at its own size,
with standard ImageNet normalization, not the custom model's input format.

Preprocessing, threshold and I/O names are taken directly from the model
card (huggingface.co/TensorFold/detectra-v1), not guessed: shortest-edge
resize -> center-crop -> ImageNet normalize -> sigmoid(logit), compared
against the model's own reported 0.65 operating threshold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np
import onnxruntime as ort

from config.loader import get_settings, resolve_path
from services.explainability.rag import explain

logger = logging.getLogger(__name__)


@lru_cache
def get_ensemble_session() -> ort.InferenceSession:
    settings = get_settings()
    path = resolve_path(settings.ensemble.model_onnx_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No ensemble ONNX model found at {path}. Download "
            "TensorFold/detectra-v1's model.onnx and place it there."
        )
    session = ort.InferenceSession(str(path), providers=[settings.inference.execution_provider])
    logger.info("Loaded ensemble ONNX model from %s", path)
    return session


def is_ensemble_loaded() -> bool:
    try:
        get_ensemble_session()
        return True
    except Exception:
        logger.exception("Ensemble model failed to load")
        return False


def preprocess_for_ensemble(image_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 -> (1, 3, 384, 384) float32, ready for the detectra-v1
    session: shortest edge -> 440 (bilinear) -> center-crop 384 -> [0,1] ->
    ImageNet normalize."""
    settings = get_settings().ensemble
    crop_h, crop_w = settings.crop_size

    h, w = image_bgr.shape[:2]
    scale = settings.resize_shortest_edge / min(h, w)
    new_w, new_h = max(round(w * scale), crop_w), max(round(h * scale), crop_h)
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    top = (new_h - crop_h) // 2
    left = (new_w - crop_w) // 2
    cropped = resized[top : top + crop_h, left : left + crop_w]

    rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array(settings.image_mean, dtype=np.float32)
    std = np.array(settings.image_std, dtype=np.float32)
    normalized = (rgb - mean) / std

    chw = np.transpose(normalized, (2, 0, 1))
    return chw[np.newaxis, ...].astype(np.float32)


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


@dataclass
class SecondOpinion:
    label: str  # "ai_generated" or "real"
    confidence: float
    explanation: str
    model_name: str = "TensorFold/detectra-v1"


def predict_second_opinion(image_bgr: np.ndarray) -> SecondOpinion:
    settings = get_settings().ensemble
    session = get_ensemble_session()

    input_array = preprocess_for_ensemble(image_bgr)
    outputs = session.run([settings.output_name], {settings.input_name: input_array})
    logit = float(np.asarray(outputs[0]).reshape(-1)[0])
    prob_ai_generated = _sigmoid(logit)

    if prob_ai_generated >= settings.ai_generated_threshold:
        return SecondOpinion(
            label="ai_generated",
            confidence=round(prob_ai_generated, 4),
            explanation=explain("ai_generated", modality="image"),
        )
    return SecondOpinion(
        label="real",
        confidence=round(1.0 - prob_ai_generated, 4),
        explanation="No AI-generation artifacts detected by this second-opinion model.",
    )
