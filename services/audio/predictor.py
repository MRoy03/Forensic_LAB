"""POST /predict/audio orchestration: decode -> feature-normalize -> one
ONNX forward pass -> label + confidence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config.loader import get_settings
from services.audio.extractor import decode_audio_bytes
from services.audio.loader import get_audio_session
from services.explainability.rag import explain

# Caps compute/memory for a single CPU forward pass; wav2vec2 handles longer
# clips, but this is a reasonable ceiling for free-tier serving hardware.
_MAX_SECONDS = 30


def _zero_mean_unit_var_norm(waveform: np.ndarray) -> np.ndarray:
    """Matches Wav2Vec2FeatureExtractor's do_normalize=True behavior
    (huggingface.co/ai8shiro/deepfake-audio-wav2vec2-ONNX's preprocessor_config.json)."""
    return (waveform - waveform.mean()) / np.sqrt(waveform.var() + 1e-7)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


@dataclass
class AudioPrediction:
    label: str  # "real" or "fake"
    confidence: float
    explanation: str


def predict_audio(audio_bytes: bytes) -> AudioPrediction:
    settings = get_settings().audio
    session = get_audio_session()

    waveform = decode_audio_bytes(audio_bytes, settings.sample_rate)
    max_samples = _MAX_SECONDS * settings.sample_rate
    if len(waveform) > max_samples:
        waveform = waveform[:max_samples]

    normalized = _zero_mean_unit_var_norm(waveform)
    input_values = normalized[np.newaxis, :].astype(np.float32)

    outputs = session.run([settings.output_name], {settings.input_name: input_values})
    logits = np.asarray(outputs[0]).reshape(-1)
    probs = _softmax(logits)

    label_idx = int(np.argmax(probs))
    label = settings.id2label[label_idx]
    confidence = float(probs[label_idx])

    return AudioPrediction(
        label=label,
        confidence=round(confidence, 4),
        explanation=explain(label, modality="audio"),
    )
