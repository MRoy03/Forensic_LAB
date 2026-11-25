"""Loads the audio deepfake-detection ONNX model (ai8shiro/deepfake-audio-
wav2vec2-ONNX). Same lru_cache singleton pattern as services/detection/loader.py."""

from __future__ import annotations

import logging
from functools import lru_cache

import onnxruntime as ort

from config.loader import get_settings, is_cached, resolve_path

logger = logging.getLogger(__name__)


@lru_cache
def get_audio_session() -> ort.InferenceSession:
    settings = get_settings()
    path = resolve_path(settings.audio.model_onnx_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No audio ONNX model found at {path}. Download "
            "ai8shiro/deepfake-audio-wav2vec2-ONNX's model.onnx and place it there."
        )
    session = ort.InferenceSession(str(path), providers=[settings.inference.execution_provider])
    logger.info("Loaded audio ONNX model from %s", path)
    return session


def is_audio_model_loaded() -> bool:
    """Passive check: has this already been loaded, without triggering a
    load as a side effect -- only ever called from /health, which must
    never force this (the largest) model into memory just by being polled."""
    return is_cached(get_audio_session)
