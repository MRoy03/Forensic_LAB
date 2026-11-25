"""API routes: thin wrappers that validate input, delegate to services, and
translate exceptions into proper HTTP status codes.

Split into four routers so a deployment can enable only what it actually
serves -- the audio model and Organika are the two largest models (see
DEPLOYMENT.md's memory-budget section), and on memory-constrained free
hosts are meant to run as their OWN separate service(s) rather than
forcing every model into one container:
  - `health_router`: /health, always included, but only probes the
    loaders for subsystems this deployment actually has enabled -- using
    each one's PASSIVE is_X_loaded() check (never triggers a load as a
    side effect of merely being polled; see config.loader.is_cached).
  - `router`: everything else "core" (image/video/playground/feedback) --
    Organika's second-opinion is included here too if ORGANIKA_ENABLED,
    but can be switched off independently of the rest of core (e.g. on
    Render, to keep its memory budget to detectra-v1 alone).
  - `audio_router`: just /predict/audio.
  - `organika_router`: just /predict/organika, a standalone endpoint for
    deployments that serve Organika on its own (e.g. the Oracle VM,
    alongside /predict/audio) -- the frontend calls this directly, the
    same pattern already used for the separate audio deployment, rather
    than one backend calling another backend over HTTP.
app/main.py reads CORE_ENABLED/AUDIO_ENABLED/ORGANIKA_ENABLED from the
environment and includes routers accordingly; all default to true, so
local dev and the original single-container deployment behave exactly as
before."""

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
    SecondOpinion,
)
from config.loader import get_settings
from services.audio.predictor import predict_audio
from services.detection.ensemble import ORGANIKA_ENABLED, predict_organika
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
organika_router = APIRouter()

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

    organika_loaded = False
    if ORGANIKA_ENABLED:
        from services.detection.ensemble import is_organika_loaded

        organika_loaded = is_organika_loaded()

    audio_loaded = False
    if AUDIO_ENABLED:
        from services.audio.loader import is_audio_model_loaded

        audio_loaded = is_audio_model_loaded()

    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        runtime=settings.inference.runtime,
        ensemble_model_loaded=ensemble_loaded,
        organika_model_loaded=organika_loaded,
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


@organika_router.post("/predict/organika", response_model=SecondOpinion)
async def predict_organika_route(file: UploadFile = File(...)) -> SecondOpinion:
    """Standalone Organika/sdxl-detector endpoint, for deployments that
    serve it on its own rather than bundled into /predict/image (e.g. the
    Oracle VM) -- the frontend calls this directly alongside /predict/image
    and merges both into one set of second opinions, same pattern as the
    separate audio deployment."""
    data = await file.read()
    try:
        image_bgr = decode_image_bytes(data)
        opinion = predict_organika(image_bgr)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("predict_organika failed")
        raise HTTPException(status_code=500, detail="Internal error during Organika prediction") from exc

    return SecondOpinion(
        label=opinion.label,
        confidence=opinion.confidence,
        model_name=opinion.model_name,
        explanation=opinion.explanation,
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
