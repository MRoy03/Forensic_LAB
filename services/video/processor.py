"""POST /predict/video orchestration: extract -> sample -> classify each
sampled frame -> aggregate into a single verdict."""

from __future__ import annotations

import tempfile
from pathlib import Path

from app.schemas import FrameResult, Label, PredictVideoResponse
from config.loader import get_settings
from services.detection.classifier import interpret_classification
from services.detection.loader import get_session
from services.preprocessing.pipeline import preprocess_image_batch
from services.video.aggregator import aggregate_frame_results
from services.video.extractor import extract_frames, get_frame_count, get_video_fps
from services.video.sampler import sample_frame_indices


def _classify_frame(frame_bgr, session, settings) -> tuple[str, float]:
    input_array = preprocess_image_batch(frame_bgr, tuple(settings.inference.image_size))
    outputs = session.run(
        ["cls_logits", "manip_logits"],
        {settings.inference.input_name: input_array},
    )
    result = interpret_classification(outputs[0], outputs[1], settings)
    return result.label, result.confidence


def process_video(video_bytes: bytes) -> PredictVideoResponse:
    settings = get_settings()
    session = get_session()

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        source_fps = get_video_fps(tmp_path)
        total_frames = get_frame_count(tmp_path)
        sampled_indices = set(
            sample_frame_indices(
                total_frames,
                source_fps,
                settings.video.sample_fps,
                max_frames=settings.video.max_frames_to_process,
            )
        )

        frame_results: list[FrameResult] = []
        for index, frame in extract_frames(tmp_path):
            if index not in sampled_indices:
                continue
            label, confidence = _classify_frame(frame, session, settings)
            frame_results.append(
                FrameResult(
                    frame_index=index,
                    timestamp_sec=round(index / source_fps, 3),
                    label=Label(label),
                    confidence=round(confidence, 4),
                )
            )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    aggregated = aggregate_frame_results(frame_results, settings.video.fake_frame_ratio_threshold)

    explanation = (
        f"{aggregated['fake_frames']} of {len(frame_results)} sampled frames flagged as fake "
        f"({aggregated['fake_frame_ratio']:.0%}); threshold is "
        f"{settings.video.fake_frame_ratio_threshold:.0%}."
    )

    return PredictVideoResponse(
        label=Label(aggregated["label"]),
        confidence=aggregated["confidence"],
        fake_frame_ratio=aggregated["fake_frame_ratio"],
        total_frames_sampled=len(frame_results),
        fake_frames=aggregated["fake_frames"],
        frame_results=frame_results,
        explanation=explanation,
    )
