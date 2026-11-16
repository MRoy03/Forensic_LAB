"""Shared pytest fixtures: synthetic sample image/video bytes, generated on
the fly so no binary fixtures need to be committed."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def isolate_feedback_storage(tmp_path, monkeypatch):
    """Without this, every test run appends real rows into the project's
    actual data/feedback_data/feedback_log.csv -- tests/test_feedback.py's
    hardcoded payloads accumulated there silently for weeks before this was
    caught, masquerading as genuine user feedback."""
    import services.feedback.storage as storage_module

    monkeypatch.setattr(storage_module, "resolve_path", lambda p: tmp_path / p)


@pytest.fixture(scope="session")
def client() -> TestClient:
    from app.main import app

    return TestClient(app)


@pytest.fixture
def sample_image_bytes() -> bytes:
    rng = np.random.default_rng(42)
    image = rng.integers(0, 255, size=(120, 160, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


@pytest.fixture
def sample_video_bytes(tmp_path) -> bytes:
    path = tmp_path / "sample.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (160, 120))

    rng = np.random.default_rng(7)
    for _ in range(20):
        frame = rng.integers(0, 255, size=(120, 160, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()

    return path.read_bytes()
