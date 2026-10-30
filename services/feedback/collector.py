"""POST /feedback orchestration: builds a feedback record and stores it."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from app.schemas import FeedbackRequest
from config.loader import get_settings
from services.feedback.storage import save_feedback
from services.postprocessing.formatter import base64_png_to_array
import cv2


def submit_feedback(payload: FeedbackRequest) -> str:
    settings = get_settings()
    feedback_id = str(uuid.uuid4())

    image_bytes = None
    if payload.image_base64:
        image_array = base64_png_to_array(payload.image_base64)
        ok, encoded = cv2.imencode(".png", image_array)
        if ok:
            image_bytes = encoded.tobytes()

    record = {
        "feedback_id": feedback_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "feedback_type": payload.feedback_type,
        "predicted_label": payload.predicted_label.value,
        "predicted_manipulation_type": payload.predicted_manipulation_type.value,
        "is_correct": payload.is_correct,
        "correct_label": payload.correct_label.value if payload.correct_label else "",
        "correct_manipulation_type": (
            payload.correct_manipulation_type.value if payload.correct_manipulation_type else ""
        ),
        "comments": payload.comments or "",
    }

    save_feedback(
        record,
        image_bytes,
        log_csv=settings.feedback.log_csv,
        images_dir=settings.feedback.images_dir,
    )

    return feedback_id
