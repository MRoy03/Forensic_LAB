"""Dispatches to the right synthetic-manipulation generator by type name."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config.loader import get_settings
from services.playground.beautify import apply_beautify
from services.playground.blur import apply_blur
from services.playground.compression import apply_compression_artifact
from services.playground.copy_move import apply_copy_move
from services.playground.splice import apply_splice
from services.playground.upscale import apply_upscale


@dataclass
class PlaygroundResult:
    image: np.ndarray
    mask: np.ndarray | None
    label: str
    manipulation_type: str


def generate(manipulation_type: str, image_bgr: np.ndarray, params: dict | None = None) -> PlaygroundResult:
    settings = get_settings()
    params = params or {}

    if manipulation_type == "none":
        return PlaygroundResult(image=image_bgr, mask=None, label="real", manipulation_type="none")

    if manipulation_type == "copy_move":
        patch_w, patch_h = settings.playground.default_patch_size
        image, mask = apply_copy_move(
            image_bgr,
            patch_size=(params.get("patch_w", patch_w), params.get("patch_h", patch_h)),
            num_patches=params.get("num_patches", 1),
        )
    elif manipulation_type == "splicing":
        image, mask = apply_splice(image_bgr)
    elif manipulation_type == "compression":
        image, mask = apply_compression_artifact(
            image_bgr, quality=params.get("quality", settings.playground.default_jpeg_quality)
        )
    elif manipulation_type == "blur":
        image, mask = apply_blur(image_bgr, ksize=params.get("ksize", settings.playground.default_blur_ksize))
    elif manipulation_type == "beautify":
        image, mask = apply_beautify(image_bgr)
    elif manipulation_type == "upscale":
        image, mask = apply_upscale(
            image_bgr, scale=params.get("scale", settings.playground.default_upscale_factor)
        )
    else:
        raise ValueError(f"Unknown manipulation_type: {manipulation_type}")

    return PlaygroundResult(image=image, mask=mask, label="fake", manipulation_type=manipulation_type)
