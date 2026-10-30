"""Frame extraction from video files via OpenCV."""

from __future__ import annotations

from typing import Iterator

import cv2
import numpy as np


def get_video_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        return fps if fps and fps > 0 else 25.0
    finally:
        cap.release()


def get_frame_count(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def extract_frames(video_path: str) -> Iterator[tuple[int, np.ndarray]]:
    """Yields (frame_index, frame_bgr) for every frame in the video, in order."""
    cap = cv2.VideoCapture(video_path)
    try:
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield index, frame
            index += 1
    finally:
        cap.release()
