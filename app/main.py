"""FastAPI application entry point.

Which feature set this process actually serves is controlled by three
environment variables, all defaulting to on so local dev and the original
single-container deployment are unaffected:
  - ENABLE_CORE=false     -- skip image/video/playground/feedback entirely
  - ENABLE_AUDIO=false    -- skip /predict/audio (and never load its model)
  - ENABLE_ORGANIKA=false -- skip Organika's second opinion (both the
    standalone /predict/organika route and, independently, whether
    /predict/image includes it) -- e.g. off on Render (keeping its budget
    to detectra-v1 alone) but on alongside audio on the Oracle VM.
Lets the two largest models (audio, Organika -- see DEPLOYMENT.md) each run
as their own separate deployment on memory-constrained free hosts, instead
of every model being forced into one container. See app/routes.py for the
router split this reads."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.routes import (
    AUDIO_ENABLED,
    CORE_ENABLED,
    ORGANIKA_ENABLED,
    audio_router,
    health_router,
    organika_router,
    router,
)
from config.loader import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if CORE_ENABLED:
        from services.detection.loader import is_model_loaded

        if is_model_loaded():
            logger.info("Model loaded successfully on startup.")
        else:
            logger.warning(
                "Model failed to load on startup -- /predict endpoints will error until "
                "a valid .onnx file is available at the configured path."
            )
    yield


app = FastAPI(title=settings.app.name, version=settings.app.version, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.app.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
if CORE_ENABLED:
    app.include_router(router)
if AUDIO_ENABLED:
    app.include_router(audio_router)
if ORGANIKA_ENABLED:
    app.include_router(organika_router)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
