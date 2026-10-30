"""Interpret raw cls_logits / manip_logits arrays into label + confidence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config.loader import Settings


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


@dataclass
class ClassifierOutput:
    label: str
    confidence: float
    manipulation_type: str
    manipulation_confidence: float


def interpret_classification(
    cls_logits: np.ndarray, manip_logits: np.ndarray, settings: Settings
) -> ClassifierOutput:
    """cls_logits, manip_logits: shape (1, N) or (N,) raw model outputs for a
    single sample (batch size 1 at serving time)."""
    cls_probs = _softmax(np.asarray(cls_logits).reshape(-1))
    manip_probs = _softmax(np.asarray(manip_logits).reshape(-1))

    label_idx = int(np.argmax(cls_probs))
    label = settings.labels.binary[label_idx]
    confidence = float(cls_probs[label_idx])

    manip_idx = int(np.argmax(manip_probs))
    manipulation_type = settings.labels.manipulation_types[manip_idx]
    manipulation_confidence = float(manip_probs[manip_idx])

    if label == "real":
        manipulation_type = settings.labels.manipulation_type_none
        manipulation_confidence = 1.0

    return ClassifierOutput(
        label=label,
        confidence=confidence,
        manipulation_type=manipulation_type,
        manipulation_confidence=manipulation_confidence,
    )
