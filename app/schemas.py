"""Pydantic request/response models for every API endpoint."""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel


class Label(str, Enum):
    real = "real"
    fake = "fake"


class ManipulationType(str, Enum):
    copy_move = "copy_move"
    splicing = "splicing"
    inpainting = "inpainting"
    gan = "gan"
    compression = "compression"
    beautify = "beautify"
    none = "none"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    runtime: str
    ensemble_model_loaded: bool
    audio_model_loaded: bool
    rag_model_loaded: bool


class SecondOpinion(BaseModel):
    label: Literal["ai_generated", "real"]
    confidence: float
    model_name: str
    explanation: str


class PredictImageResponse(BaseModel):
    label: Label
    confidence: float
    manipulation_type: ManipulationType
    mask: str  # base64-encoded PNG, "" when label == real
    gradcam: str  # base64-encoded PNG (Eigen-CAM heatmap overlay)
    explanation: str
    # Independent pretrained model's own verdict (TensorFold/detectra-v1),
    # shown alongside rather than fused into the primary verdict -- stays
    # transparent when the two disagree instead of silently overriding one
    # with an unvalidated fusion heuristic. None if that model isn't loaded.
    second_opinion: Optional[SecondOpinion] = None


class PredictAudioResponse(BaseModel):
    label: Literal["real", "fake"]
    confidence: float
    explanation: str


class FrameResult(BaseModel):
    frame_index: int
    timestamp_sec: float
    label: Label
    confidence: float


class PredictVideoResponse(BaseModel):
    label: Label
    confidence: float
    fake_frame_ratio: float
    total_frames_sampled: int
    fake_frames: int
    frame_results: list[FrameResult]
    explanation: str


class PlaygroundGenerateResponse(BaseModel):
    image: str  # base64-encoded PNG
    mask: Optional[str] = None  # base64-encoded PNG, if applicable
    label: Label
    # Plain str, not ManipulationType: the playground's manipulation taxonomy
    # (e.g. "blur", "upscale") is broader than the classifier's 6 predicted
    # classes, so it isn't constrained to that enum.
    manipulation_type: str


class FeedbackRequest(BaseModel):
    feedback_type: Literal["simple", "advanced"]
    predicted_label: Label
    predicted_manipulation_type: ManipulationType
    is_correct: Optional[bool] = None
    correct_label: Optional[Label] = None
    correct_manipulation_type: Optional[ManipulationType] = None
    comments: Optional[str] = None
    image_base64: Optional[str] = None


class FeedbackResponse(BaseModel):
    status: str
    feedback_id: str
