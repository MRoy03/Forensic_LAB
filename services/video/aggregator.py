"""Aggregate per-frame classification results into a single video verdict."""

from __future__ import annotations

from app.schemas import FrameResult


def aggregate_frame_results(results: list[FrameResult], threshold: float) -> dict:
    total = len(results)
    if total == 0:
        return {
            "label": "real",
            "confidence": 0.0,
            "fake_frame_ratio": 0.0,
            "fake_frames": 0,
        }

    fake_frames = sum(1 for r in results if r.label == "fake")
    ratio = fake_frames / total
    label = "fake" if ratio > threshold else "real"

    matching = [r.confidence for r in results if r.label == label]
    confidence = sum(matching) / len(matching) if matching else 0.0

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "fake_frame_ratio": round(ratio, 4),
        "fake_frames": fake_frames,
    }
