"""Loads the exported ONNX model for serving. No torch dependency -- this is
the only place the live app touches the model, and it only ever talks to
onnxruntime."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import onnxruntime as ort

from config.loader import get_settings, resolve_path

logger = logging.getLogger(__name__)


def build_session(onnx_path: str | Path | None = None) -> ort.InferenceSession:
    settings = get_settings()
    path = resolve_path(onnx_path or settings.models.image_model_onnx_path)

    if not path.exists():
        raise FileNotFoundError(
            f"No ONNX model found at {path}. Run "
            "`python services/detection/export_onnx.py --output "
            f"{settings.models.image_model_onnx_path}` (requires requirements-train.txt) "
            "to produce one, or export a trained model from the Colab notebooks."
        )

    session = ort.InferenceSession(str(path), providers=[settings.inference.execution_provider])
    logger.info("Loaded ONNX model from %s with providers=%s", path, session.get_providers())
    return session


@lru_cache
def get_session() -> ort.InferenceSession:
    return build_session()


def is_model_loaded() -> bool:
    try:
        get_session()
        return True
    except Exception:
        logger.exception("Model failed to load")
        return False
