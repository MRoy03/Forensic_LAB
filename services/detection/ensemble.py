"""Two pretrained "second opinion" image models (TensorFold/detectra-v1 and
Organika/sdxl-detector), each loaded and run independently of the custom
5-channel RGB+ELA+FFT model. Both expect plain 3-channel RGB at their own
size with standard ImageNet normalization, not the custom model's input
format -- but their preprocessing and output heads differ from each other
too (384 vs 224 input, shortest-edge+crop vs plain resize, sigmoid vs
2-class softmax), so each gets its own preprocessing/predict pair rather
than a shared abstraction forced over two genuinely different models.

detectra-v1: preprocessing/threshold/I-O names taken directly from its
model card (huggingface.co/TensorFold/detectra-v1) -- shortest-edge resize
-> center-crop -> ImageNet normalize -> sigmoid(logit), compared against
its own reported 0.65 operating threshold.

Organika/sdxl-detector: converted from its raw PyTorch safetensors weights
via scripts/convert_organika_to_onnx.py (that repo ships no ONNX export of
its own). Preprocessing/label order taken from its config.json/
preprocessor_config.json -- plain resize to 224x224 (no crop step), 2-class
softmax with id2label {0: "artificial", 1: "human"}.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np
import onnxruntime as ort

from config.loader import get_settings, is_cached, resolve_path
from services.explainability.rag import explain

logger = logging.getLogger(__name__)

# Independent of CORE_ENABLED (app/routes.py) -- lets Organika specifically
# be disabled on one deployment (e.g. Render, to keep its memory budget to
# detectra-v1 alone) while running standalone elsewhere (e.g. alongside the
# audio model on the Oracle VM, via its own /predict/organika route) --
# see app/routes.py's organika_router.
ORGANIKA_ENABLED = os.environ.get("ENABLE_ORGANIKA", "true").lower() == "true"


def _no_arena_session_options() -> ort.SessionOptions:
    """enable_cpu_mem_arena=False: onnxruntime's default memory arena grows
    to fit the largest input seen so far and never releases it back --
    confirmed via real kernel OOM-killer logs on the Oracle VM (climbing
    RSS across repeated kills on the audio model specifically, the
    worst-affected due to variable-length input). Applied everywhere for
    consistency/safety, not just there -- costs a little speed (plain
    malloc/free per allocation instead of a reused arena) for bounded,
    predictable memory on every memory-constrained free host this project
    runs on."""
    sess_options = ort.SessionOptions()
    sess_options.enable_cpu_mem_arena = False
    return sess_options


@lru_cache
def get_ensemble_session() -> ort.InferenceSession:
    settings = get_settings()
    path = resolve_path(settings.ensemble.model_onnx_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No ensemble ONNX model found at {path}. Download "
            "TensorFold/detectra-v1's model.onnx and place it there."
        )
    session = ort.InferenceSession(
        str(path), sess_options=_no_arena_session_options(), providers=[settings.inference.execution_provider]
    )
    logger.info("Loaded ensemble ONNX model from %s", path)
    return session


def is_ensemble_available() -> bool:
    """Active check: does the model exist and load successfully? Triggers
    loading if not already cached -- used to gate whether
    predict_second_opinions() actually attempts this model for a real
    prediction request, where an answer is genuinely needed."""
    try:
        get_ensemble_session()
        return True
    except Exception:
        logger.exception("Ensemble model failed to load")
        return False


def is_ensemble_loaded() -> bool:
    """Passive check: has this already been loaded, without triggering a
    load as a side effect -- for /health, which must never force a heavy
    model into memory just by being polled."""
    return is_cached(get_ensemble_session)


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


def predict_detectra(image_bgr: np.ndarray) -> SecondOpinion:
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
            model_name="TensorFold/detectra-v1",
        )
    return SecondOpinion(
        label="real",
        confidence=round(1.0 - prob_ai_generated, 4),
        explanation="No AI-generation artifacts detected by this second-opinion model.",
        model_name="TensorFold/detectra-v1",
    )


@lru_cache
def get_organika_session() -> ort.InferenceSession:
    settings = get_settings()
    path = resolve_path(settings.organika.model_onnx_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No Organika ONNX model found at {path}. Run "
            "scripts/convert_organika_to_onnx.py to produce one from "
            "models/pending/organika_sdxl_detector.safetensors."
        )
    session = ort.InferenceSession(
        str(path), sess_options=_no_arena_session_options(), providers=[get_settings().inference.execution_provider]
    )
    logger.info("Loaded Organika ONNX model from %s", path)
    return session


def is_organika_available() -> bool:
    """Active check -- see is_ensemble_available()'s docstring."""
    try:
        get_organika_session()
        return True
    except Exception:
        logger.exception("Organika model failed to load")
        return False


def is_organika_loaded() -> bool:
    """Passive check -- see is_ensemble_loaded()'s docstring."""
    return is_cached(get_organika_session)


def preprocess_for_organika(image_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 -> (1, 3, 224, 224) float32: plain resize (no crop) ->
    [0,1] -> ImageNet normalize, per Organika/sdxl-detector's
    preprocessor_config.json (a fixed height/width size, not a separate
    shortest-edge + crop_size pair like detectra-v1's)."""
    settings = get_settings().organika
    h, w = settings.image_size

    resized = cv2.resize(image_bgr, (w, h), interpolation=cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array(settings.image_mean, dtype=np.float32)
    std = np.array(settings.image_std, dtype=np.float32)
    normalized = (rgb - mean) / std

    chw = np.transpose(normalized, (2, 0, 1))
    return chw[np.newaxis, ...].astype(np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def predict_organika(image_bgr: np.ndarray) -> SecondOpinion:
    settings = get_settings().organika
    session = get_organika_session()

    input_array = preprocess_for_organika(image_bgr)
    outputs = session.run([settings.output_name], {settings.input_name: input_array})
    logits = np.asarray(outputs[0]).reshape(-1)
    probs = _softmax(logits)
    prob_artificial = float(probs[0])  # id2label: 0=artificial, 1=human

    if prob_artificial >= 0.5:
        return SecondOpinion(
            label="ai_generated",
            confidence=round(prob_artificial, 4),
            explanation=explain("ai_generated", modality="image"),
            model_name="Organika/sdxl-detector",
        )
    return SecondOpinion(
        label="real",
        confidence=round(1.0 - prob_artificial, 4),
        explanation="No AI-generation artifacts detected by this second-opinion model.",
        model_name="Organika/sdxl-detector",
    )


def predict_second_opinions(image_bgr: np.ndarray) -> list[SecondOpinion]:
    """Runs every available, enabled second-opinion model, skipping any
    that aren't rather than failing the whole prediction -- each one is an
    independent, optional signal, not a required part of the pipeline.
    Organika is checked against ORGANIKA_ENABLED too, not just whether its
    model file happens to be present -- lets a deployment (e.g. Render)
    deliberately not run it locally even if the file is there, because it's
    served from a separate standalone deployment instead (e.g. the Oracle
    VM's /predict/organika, called directly by the frontend)."""
    opinions = []
    candidates = [(is_ensemble_available, predict_detectra)]
    if ORGANIKA_ENABLED:
        candidates.append((is_organika_available, predict_organika))
    for is_available, predict in candidates:
        if not is_available():
            continue
        try:
            opinions.append(predict(image_bgr))
        except Exception:
            logger.exception("Second-opinion prediction failed for %s", predict.__name__)
    return opinions
