"""Main orchestration for POST /predict/image: decode -> preprocess -> one
ONNX forward pass -> classification + mask + Eigen-CAM -> response."""

from __future__ import annotations

from app.schemas import Label, ManipulationType, PredictImageResponse
from config.loader import get_settings
from services.detection.classifier import interpret_classification
from services.detection.loader import get_session
from services.detection.segmentor import interpret_mask
from services.explainability.gradcam import generate_eigencam
from services.postprocessing.formatter import array_to_base64_png
from services.postprocessing.overlay import overlay_heatmap, overlay_mask
from services.preprocessing.pipeline import decode_image_bytes, preprocess_image_batch

_EXPLANATIONS = {
    "copy_move": "Duplicated region patterns detected, consistent with copy-move forgery.",
    "splicing": "Edge and lighting inconsistencies detected at a region boundary, consistent with splicing.",
    "inpainting": "Texture discontinuity detected, consistent with inpainting/content-aware fill.",
    "gan": "Statistical artifacts consistent with GAN-based generation detected.",
    "compression": "Localized compression inconsistencies detected.",
    "beautify": "Skin-smoothing / beautification artifacts detected.",
    "none": "No manipulation artifacts detected; image appears authentic.",
}


def _explanation_for(manipulation_type: str) -> str:
    return _EXPLANATIONS.get(manipulation_type, "Manipulation artifacts detected.")


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
        explanation=_explanation_for(result.manipulation_type),
    )
