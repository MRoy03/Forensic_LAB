"""API routes: thin wrappers that validate input, delegate to services, and
translate exceptions into proper HTTP status codes.

Split into three routers so a deployment can enable only what it actually
serves -- the audio model is by far the largest (see DEPLOYMENT.md's
memory-budget section) and, on memory-constrained free hosts, is meant to
run as its OWN separate service rather than forcing every model into one
container:
  - `health_router`: /health, always included, but only probes the
    loaders for subsystems this deployment actually has enabled.
  - `router`: everything "core" (image/video/playground/feedback).
  - `audio_router`: just /predict/audio.
app/main.py reads CORE_ENABLED/AUDIO_ENABLED from the environment and
includes routers accordingly; both default to true, so local dev and the
existing single-container deployment behave exactly as before."""

from __future__ import annotations

import json
import logging
import os

from fastapi import APIRouter, Form, HTTPException, UploadFile, File

from app.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    PlaygroundGenerateResponse,
    PredictAudioResponse,
    PredictImageResponse,
    PredictVideoResponse,
)
from config.loader import get_settings
from services.audio.predictor import predict_audio
from services.detection.predictor import predict_image
from services.feedback.collector import submit_feedback
from services.playground.controller import generate as playground_generate
from services.postprocessing.formatter import array_to_base64_png
from services.preprocessing.pipeline import decode_image_bytes
from services.video.processor import process_video

logger = logging.getLogger(__name__)
health_router = APIRouter()
router = APIRouter()
audio_router = APIRouter()

# Also read directly by app/main.py to decide which routers to include.
CORE_ENABLED = os.environ.get("ENABLE_CORE", "true").lower() == "true"
AUDIO_ENABLED = os.environ.get("ENABLE_AUDIO", "true").lower() == "true"


@health_router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()

    model_loaded = ensemble_loaded = rag_loaded = False
    if CORE_ENABLED:
        from services.detection.ensemble import is_ensemble_loaded
        from services.detection.loader import is_model_loaded
        from services.explainability.rag import is_rag_loaded

        model_loaded = is_model_loaded()
        ensemble_loaded = is_ensemble_loaded()
        rag_loaded = is_rag_loaded()

    audio_loaded = False
    if AUDIO_ENABLED:
        from services.audio.loader import is_audio_model_loaded

        audio_loaded = is_audio_model_loaded()

    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        runtime=settings.inference.runtime,
        ensemble_model_loaded=ensemble_loaded,
        audio_model_loaded=audio_loaded,
        rag_model_loaded=rag_loaded,
    )


@router.post("/predict/image", response_model=PredictImageResponse)
async def predict_image_route(file: UploadFile = File(...)) -> PredictImageResponse:
    data = await file.read()
    try:
        return predict_image(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("predict_image failed")
        raise HTTPException(status_code=500, detail="Internal error during image prediction") from exc


@router.post("/predict/video", response_model=PredictVideoResponse)
async def predict_video_route(file: UploadFile = File(...)) -> PredictVideoResponse:
    data = await file.read()
    try:
        return process_video(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("predict_video failed")
        raise HTTPException(status_code=500, detail="Internal error during video prediction") from exc


@audio_router.post("/predict/audio", response_model=PredictAudioResponse)
async def predict_audio_route(file: UploadFile = File(...)) -> PredictAudioResponse:
    data = await file.read()
    try:
        result = predict_audio(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("predict_audio failed")
        raise HTTPException(status_code=500, detail="Internal error during audio prediction") from exc

    return PredictAudioResponse(
        label=result.label, confidence=result.confidence, explanation=result.explanation
    )


@router.post("/playground/generate", response_model=PlaygroundGenerateResponse)
async def playground_generate_route(
    manipulation_type: str = Form(...),
    file: UploadFile = File(...),
    params: str | None = Form(None),
) -> PlaygroundGenerateResponse:
    data = await file.read()
    try:
        image = decode_image_bytes(data)
        parsed_params = json.loads(params) if params else {}
        result = playground_generate(manipulation_type, image, parsed_params)
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("playground_generate failed")
        raise HTTPException(status_code=500, detail="Internal error during playground generation") from exc

    return PlaygroundGenerateResponse(
        image=array_to_base64_png(result.image),
        mask=array_to_base64_png(result.mask) if result.mask is not None else None,
        label=result.label,
        manipulation_type=result.manipulation_type,
    )


@router.post("/feedback", response_model=FeedbackResponse)
def feedback_route(payload: FeedbackRequest) -> FeedbackResponse:
    try:
        feedback_id = submit_feedback(payload)
    except Exception as exc:
        logger.exception("submit_feedback failed")
        raise HTTPException(status_code=500, detail="Internal error while storing feedback") from exc

    return FeedbackResponse(status="ok", feedback_id=feedback_id)
