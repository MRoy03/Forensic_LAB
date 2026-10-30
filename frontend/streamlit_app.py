"""ForensicAI Lab -- Streamlit frontend.

Talks to the FastAPI backend over HTTP; has no model/ML code of its own.
Run with: streamlit run frontend/streamlit_app.py
(Run `uvicorn app.main:app` separately first.)
"""

from __future__ import annotations

import base64
import io
import random
import sys
from pathlib import Path

import requests
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.loader import get_settings  # noqa: E402

settings = get_settings()
API_BASE_URL = settings.api.base_url

MANIPULATION_TYPES = ["copy_move", "splicing", "compression", "blur", "beautify", "upscale"]
CLASSIFIER_MANIPULATION_TYPES = settings.labels.manipulation_types + [
    settings.labels.manipulation_type_none
]


def b64_to_image(b64_string: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_string)))


def image_to_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def render_predict_result(body: dict) -> None:
    cols = st.columns(3)
    with cols[0]:
        st.caption("Prediction")
        st.metric("Label", body["label"].upper(), delta=f"{body['confidence']:.0%} confidence")
        st.write(f"**Manipulation type:** {body['manipulation_type']}")
        st.write(body["explanation"])
    with cols[1]:
        st.caption("Tampered-region mask")
        if body.get("mask"):
            st.image(b64_to_image(body["mask"]), use_container_width=True)
        else:
            st.info("No mask (image predicted real).")
    with cols[2]:
        st.caption("Eigen-CAM heatmap")
        st.image(b64_to_image(body["gradcam"]), use_container_width=True)
        st.caption(
            "Heatmap quality depends on trained weights -- expect generic/noisy "
            "saliency until a trained model is exported (see models/README.md)."
        )


def page_image_detection() -> None:
    st.header("Image Detection")
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded is None:
        return

    st.image(uploaded, caption="Uploaded image", width=300)
    if st.button("Analyze image"):
        with st.spinner("Analyzing..."):
            response = requests.post(
                f"{API_BASE_URL}/predict/image",
                files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                timeout=60,
            )
        if response.status_code != 200:
            st.error(f"Request failed: {response.status_code} {response.text}")
            return
        render_predict_result(response.json())


def page_video_detection() -> None:
    st.header("Video Detection")
    uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded is None:
        return

    st.video(uploaded)
    if st.button("Analyze video"):
        with st.spinner("Extracting and analyzing frames..."):
            response = requests.post(
                f"{API_BASE_URL}/predict/video",
                files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                timeout=300,
            )
        if response.status_code != 200:
            st.error(f"Request failed: {response.status_code} {response.text}")
            return

        body = response.json()
        st.metric("Verdict", body["label"].upper(), delta=f"{body['confidence']:.0%} confidence")
        st.write(
            f"{body['fake_frames']} of {body['total_frames_sampled']} sampled frames "
            f"flagged fake ({body['fake_frame_ratio']:.0%})."
        )
        st.write(body["explanation"])
        st.dataframe(body["frame_results"], use_container_width=True)


def _generate_playground_image(manipulation_type: str, image_bytes: bytes, filename: str) -> dict | None:
    response = requests.post(
        f"{API_BASE_URL}/playground/generate",
        data={"manipulation_type": manipulation_type},
        files={"file": (filename, image_bytes, "image/png")},
        timeout=60,
    )
    if response.status_code != 200:
        st.error(f"Request failed: {response.status_code} {response.text}")
        return None
    return response.json()


def page_playground() -> None:
    st.header("Playground -- Synthetic Manipulation Generator")
    uploaded = st.file_uploader("Upload a source image", type=["jpg", "jpeg", "png", "bmp"], key="playground_upload")
    manipulation_type = st.selectbox("Manipulation type", MANIPULATION_TYPES)

    if uploaded is None:
        return

    st.image(uploaded, caption="Source image", width=300)
    if st.button("Generate"):
        with st.spinner("Generating..."):
            body = _generate_playground_image(manipulation_type, uploaded.getvalue(), uploaded.name)
        if body is None:
            return

        cols = st.columns(2)
        with cols[0]:
            st.caption("Manipulated image")
            st.image(b64_to_image(body["image"]), use_container_width=True)
        with cols[1]:
            st.caption("Ground-truth mask")
            if body.get("mask"):
                st.image(b64_to_image(body["mask"]), use_container_width=True)
            else:
                st.info("This manipulation type doesn't produce a localized mask.")


def page_quiz() -> None:
    st.header("Quiz Mode")
    st.write("Upload an image; we'll randomly apply (or skip) a manipulation and test your eye.")

    uploaded = st.file_uploader("Upload a source image", type=["jpg", "jpeg", "png", "bmp"], key="quiz_upload")
    if uploaded is None:
        return

    if st.button("New challenge") or "quiz_challenge" not in st.session_state:
        challenge_type = random.choice(MANIPULATION_TYPES + ["none"])
        with st.spinner("Preparing challenge..."):
            body = _generate_playground_image(challenge_type, uploaded.getvalue(), uploaded.name)
        if body is None:
            return
        st.session_state["quiz_challenge"] = body
        st.session_state.pop("quiz_revealed", None)

    challenge = st.session_state["quiz_challenge"]
    st.image(b64_to_image(challenge["image"]), caption="Guess: real or fake?", width=350)

    guess_label = st.radio("Your guess", ["real", "fake"], horizontal=True)
    guess_type = st.selectbox("If fake, which type?", CLASSIFIER_MANIPULATION_TYPES)

    if st.button("Reveal answer"):
        st.session_state["quiz_revealed"] = True

    if st.session_state.get("quiz_revealed"):
        st.subheader("Ground truth")
        st.write(f"**Actual:** {challenge['label']} ({challenge['manipulation_type']})")
        correct = guess_label == challenge["label"]
        st.success("Correct!") if correct else st.error("Not quite.")

        with st.spinner("Running AI prediction..."):
            img_bytes = base64.b64decode(challenge["image"])
            ai_response = requests.post(
                f"{API_BASE_URL}/predict/image",
                files={"file": ("quiz.png", img_bytes, "image/png")},
                timeout=60,
            )
        if ai_response.status_code == 200:
            st.subheader("AI prediction")
            render_predict_result(ai_response.json())


def page_feedback() -> None:
    st.header("Feedback")
    st.write("Help improve the model by reporting prediction mistakes.")

    predicted_label = st.selectbox("Predicted label", ["real", "fake"])
    predicted_type = st.selectbox("Predicted manipulation type", CLASSIFIER_MANIPULATION_TYPES)
    feedback_mode = st.radio("Feedback type", ["simple", "advanced"], horizontal=True)

    payload = {
        "feedback_type": feedback_mode,
        "predicted_label": predicted_label,
        "predicted_manipulation_type": predicted_type,
    }

    if feedback_mode == "simple":
        is_correct = st.radio("Was this correct?", ["Correct", "Incorrect"], horizontal=True)
        payload["is_correct"] = is_correct == "Correct"
    else:
        payload["correct_label"] = st.selectbox("Correct label", ["real", "fake"])
        payload["correct_manipulation_type"] = st.selectbox(
            "Correct manipulation type", CLASSIFIER_MANIPULATION_TYPES, key="correct_type"
        )
        payload["comments"] = st.text_area("Comments")

    if st.button("Submit feedback"):
        response = requests.post(f"{API_BASE_URL}/feedback", json=payload, timeout=30)
        if response.status_code == 200:
            st.success(f"Thanks! Feedback ID: {response.json()['feedback_id']}")
        else:
            st.error(f"Submission failed: {response.status_code} {response.text}")


def page_about() -> None:
    st.header("About ForensicAI Lab")
    st.markdown(
        """
        A modular AI platform for image/video forgery detection: real/fake
        classification, manipulation-type detection, tampered-region
        localization, Eigen-CAM explainability, video forgery analysis, a
        synthetic-manipulation playground, an educational quiz mode, and a
        feedback/retraining loop.

        See `README.md` for the full architecture, training roadmap, and
        resource-constraint notes (ONNX-only serving, Eigen-CAM in place of
        gradient-based Grad-CAM).
        """
    )


def main() -> None:
    st.set_page_config(page_title=settings.app.name, layout="wide")
    st.title(settings.app.name)

    pages = {
        "Image Detection": page_image_detection,
        "Video Detection": page_video_detection,
        "Playground": page_playground,
        "Quiz Mode": page_quiz,
        "Feedback": page_feedback,
        "About": page_about,
    }
    choice = st.sidebar.radio("Navigate", list(pages.keys()))
    pages[choice]()


if __name__ == "__main__":
    main()
