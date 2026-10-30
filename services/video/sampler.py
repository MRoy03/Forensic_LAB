"""Deterministic frame-index sampling at a target FPS."""

from __future__ import annotations


def sample_frame_indices(
    total_frames: int, source_fps: float, target_fps: float, max_frames: int | None = None
) -> list[int]:
    if total_frames <= 0:
        return []

    stride = max(1, round(source_fps / target_fps)) if target_fps > 0 else 1
    indices = list(range(0, total_frames, stride))

    if max_frames is not None and len(indices) > max_frames:
        indices = indices[:max_frames]

    return indices
