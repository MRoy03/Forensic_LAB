"""FastAPI application entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.routes import router
from config.loader import get_settings
from services.detection.loader import is_model_loaded

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
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

app.include_router(router)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
