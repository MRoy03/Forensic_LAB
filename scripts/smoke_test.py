"""End-to-end smoke test against a running instance of the app.

Usage:
    uvicorn app.main:app &
    python scripts/smoke_test.py [base_url]

Exercises every endpoint (health, predict/image, predict/video,
playground/generate, feedback) with synthetic, in-memory fixtures -- no
external sample files required.
"""

from __future__ import annotations

import sys

import cv2
import numpy as np
import requests

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"


def _sample_image_bytes() -> bytes:
    rng = np.random.default_rng(42)
    image = rng.integers(0, 255, size=(120, 160, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def _sample_video_bytes(tmp_path: str) -> bytes:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, 10.0, (160, 120))
    rng = np.random.default_rng(7)
    for _ in range(20):
        writer.write(rng.integers(0, 255, size=(120, 160, 3), dtype=np.uint8))
    writer.release()
    with open(tmp_path, "rb") as f:
        return f.read()


def check(name: str, condition: bool) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    if not condition:
        sys.exit(1)


def main() -> None:
    health = requests.get(f"{BASE_URL}/health", timeout=10)
    check("GET /health returns 200", health.status_code == 200)
    check("health reports model_loaded=True", health.json().get("model_loaded") is True)

    image_bytes = _sample_image_bytes()
    image_resp = requests.post(
        f"{BASE_URL}/predict/image",
        files={"file": ("sample.jpg", image_bytes, "image/jpeg")},
        timeout=60,
    )
    check("POST /predict/image returns 200", image_resp.status_code == 200)
    body = image_resp.json()
    check("predict/image has gradcam", len(body.get("gradcam", "")) > 0)

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        video_path = tmp.name
    video_bytes = _sample_video_bytes(video_path)
    video_resp = requests.post(
        f"{BASE_URL}/predict/video",
        files={"file": ("sample.mp4", video_bytes, "video/mp4")},
        timeout=120,
    )
    check("POST /predict/video returns 200", video_resp.status_code == 200)
    check("predict/video sampled frames > 0", video_resp.json().get("total_frames_sampled", 0) > 0)

    playground_resp = requests.post(
        f"{BASE_URL}/playground/generate",
        data={"manipulation_type": "copy_move"},
        files={"file": ("sample.jpg", image_bytes, "image/jpeg")},
        timeout=30,
    )
    check("POST /playground/generate returns 200", playground_resp.status_code == 200)

    feedback_resp = requests.post(
        f"{BASE_URL}/feedback",
        json={
            "feedback_type": "simple",
            "predicted_label": "fake",
            "predicted_manipulation_type": "splicing",
            "is_correct": False,
        },
        timeout=30,
    )
    check("POST /feedback returns 200", feedback_resp.status_code == 200)

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
