"""Typed access to config/settings.yaml. Single source of truth for paths/thresholds."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"


class AppSettings(BaseModel):
    name: str
    version: str
    cors_allow_origins: list[str]


class ApiSettings(BaseModel):
    base_url: str


class InferenceSettings(BaseModel):
    runtime: str
    execution_provider: str
    image_size: tuple[int, int]
    input_channels: int
    mask_threshold: float
    input_name: str
    output_names: list[str]


class ModelsSettings(BaseModel):
    image_model_onnx_path: str
    video_model_onnx_path: str
    final_model_onnx_path: str
    image_model_checkpoint: str
    video_model_checkpoint: str
    final_model_checkpoint: str
    backbone: str
    pretrained_encoder: bool


class LabelsSettings(BaseModel):
    binary: list[str]
    manipulation_types: list[str]
    manipulation_type_none: str


class VideoSettings(BaseModel):
    sample_fps: float
    fake_frame_ratio_threshold: float
    max_frames_to_process: int


class PlaygroundSettings(BaseModel):
    default_patch_size: tuple[int, int]
    default_jpeg_quality: int
    default_blur_ksize: int
    default_upscale_factor: int


class DatasetEntry(BaseModel):
    name: str
    type: str
    local_subdir: str


class DataSettings(BaseModel):
    root_local: str
    root_drive: str
    use_drive_if_available: bool
    images_dir: str
    masks_dir: str
    labels_csv: str
    synthetic_dir: str
    feedback_dir: str
    datasets_registry: list[DatasetEntry]


class FeedbackSettings(BaseModel):
    log_csv: str
    images_dir: str
    retrain_manifest: str


class LoggingSettings(BaseModel):
    level: str


class Settings(BaseModel):
    app: AppSettings
    api: ApiSettings
    inference: InferenceSettings
    models: ModelsSettings
    labels: LabelsSettings
    video: VideoSettings
    playground: PlaygroundSettings
    data: DataSettings
    feedback: FeedbackSettings
    logging: LoggingSettings


def load_settings(path: str | Path = DEFAULT_SETTINGS_PATH) -> Settings:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Settings.model_validate(raw)


@lru_cache
def get_settings() -> Settings:
    return load_settings()


def resolve_path(relative_path: str) -> Path:
    """Resolve a path from settings.yaml relative to the project root."""
    p = Path(relative_path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def resolve_dataset_root(settings: Settings | None = None) -> Path:
    """Prefer the Google-Drive-mounted dataset root when running in Colab,
    otherwise fall back to the local dataset root (creating it if missing)."""
    settings = settings or get_settings()
    in_colab = os.path.exists("/content")
    drive_root = Path(settings.data.root_drive)
    local_root = resolve_path(settings.data.root_local)

    if in_colab and settings.data.use_drive_if_available and drive_root.exists():
        return drive_root

    if local_root.exists():
        return local_root

    local_root.mkdir(parents=True, exist_ok=True)
    return local_root
