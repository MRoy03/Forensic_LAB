"""Main orchestration for POST /predict/image: decode -> preprocess -> one
ONNX forward pass -> classification + mask + Eigen-CAM -> response."""

from __future__ import annotations

import logging

from app.schemas import Label, ManipulationType, PredictImageResponse, SecondOpinion
from config.loader import get_settings
from services.detection.classifier import interpret_classification
from services.detection.ensemble import predict_second_opinions
from services.detection.loader import get_session
from services.detection.segmentor import interpret_mask
from services.explainability.gradcam import generate_eigencam
from services.explainability.rag import explain
from services.postprocessing.formatter import array_to_base64_png
from services.postprocessing.overlay import overlay_heatmap, overlay_mask
from services.preprocessing.pipeline import decode_image_bytes, preprocess_image_batch

logger = logging.getLogger(__name__)


def _second_opinions_for(original) -> list[SecondOpinion]:
    return [
        SecondOpinion(
            label=opinion.label,
            confidence=opinion.confidence,
            model_name=opinion.model_name,
            explanation=opinion.explanation,
        )
        for opinion in predict_second_opinions(original)
    ]


def predict_image(image_bytes: bytes) -> PredictImageResponse:
    settings = get_settings()
    session = get_session()

    original = decode_image_bytes(image_bytes)
    original_size = (original.shape[1], original.shape[0])  # (width, height)

    input_array = preprocess_image_batch(original, tuple(settings.inference.image_size))

    outputs = session.run(
        settings.inference.output_names,
        {settings.inference.input_name: input_array},
    )
    output_map = dict(zip(settings.inference.output_names, outputs))

    result = interpret_classification(
        output_map["cls_logits"], output_map["manip_logits"], settings
    )

    if result.label == "fake":
        mask = interpret_mask(
            output_map["mask_logits"], original_size, settings.inference.mask_threshold
        )
        mask_overlay = overlay_mask(original, mask)
        mask_b64 = array_to_base64_png(mask_overlay)
    else:
        mask_b64 = ""

    heatmap = generate_eigencam(output_map["feature_map"], original_size)
    gradcam_overlay = overlay_heatmap(original, heatmap)
    gradcam_b64 = array_to_base64_png(gradcam_overlay)

    return PredictImageResponse(
        label=Label(result.label),
        confidence=round(result.confidence, 4),
        manipulation_type=ManipulationType(result.manipulation_type),
        mask=mask_b64,
        gradcam=gradcam_b64,
        explanation=explain(result.manipulation_type, modality="image"),
        second_opinions=_second_opinions_for(original),
    )
