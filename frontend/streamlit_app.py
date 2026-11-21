"""ForensicAI Lab -- Streamlit frontend.

Talks to the FastAPI backend over HTTP; has no model/ML code of its own.
Run with: streamlit run frontend/streamlit_app.py
(Run `uvicorn app.main:app` separately first.)

Theming: base colors/fonts/radii live in .streamlit/config.toml (the
officially-supported mechanism -- see the bundled Streamlit theming
reference). The st.html() calls below only add what config.toml can't
express (gradient text, glow shadows, hover-reveal, scan animation) and
target .st-key-* classes from explicit `key=` params, never private
Streamlit class names.

Navigation is session_state-driven (`current_page`), not st.sidebar.radio
-- lets both the sidebar buttons AND the clickable Home cards control
which page renders, and lets the active sidebar button highlight itself
via Streamlit's own `type="primary"` rather than custom CSS.
"""

from __future__ import annotations

import base64
import io
import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path

import requests
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.loader import get_settings  # noqa: E402

settings = get_settings()
# Env override so the same code works unchanged whether the backend is
# localhost (local dev) or a deployed URL (e.g. a separate Render/HF Space
# host) -- see DEPLOYMENT.md.
API_BASE_URL = os.environ.get("API_BASE_URL", settings.api.base_url)
# The audio model is large enough to warrant its own separate deployment on
# memory-constrained free hosts (see DEPLOYMENT.md's split-services option)
# -- defaults to API_BASE_URL so a single combined deployment (or local dev)
# needs no extra configuration.
AUDIO_API_BASE_URL = os.environ.get("AUDIO_API_BASE_URL", API_BASE_URL)

MANIPULATION_TYPES = ["copy_move", "splicing", "inpainting", "compression", "blur", "beautify", "upscale"]
CLASSIFIER_MANIPULATION_TYPES = settings.labels.manipulation_types + [
    settings.labels.manipulation_type_none
]

MOTTO = "SEE THROUGH THE SURFACE."
TAGLINE = "AI-powered forgery detection, explainable and transparent -- from pixels to proof."

# (page key, icon, short card description, longer hover-reveal technical blurb)
FEATURE_GUIDE = [
    (
        "Image Detection", "🖼️",
        "Upload a photo and get a real/fake verdict, the manipulated region highlighted, and an explainability heatmap.",
        "Decodes the image and stacks RGB with Error-Level-Analysis and FFT "
        "frequency channels into one 5-channel tensor, fed through a "
        "multi-head EfficientNet-B0 trained for real/fake, manipulation-type, "
        "and pixel-level mask segmentation all at once. The heatmap comes from "
        "Eigen-CAM -- a gradient-free saliency method (SVD on the model's own "
        "feature map) chosen specifically because it needs no backward pass, "
        "so the whole thing runs on ONNX Runtime with zero PyTorch at serving time.",
    ),
    (
        "Video Detection", "🎞️",
        "Upload a clip; frames are sampled and scored individually, then combined into one verdict.",
        "Samples frames at a fixed rate via OpenCV, classifies each one with "
        "the same model as Image Detection, then aggregates by a configurable "
        "fake-frame-ratio threshold -- catches manipulation that only shows up "
        "in a handful of frames rather than the whole clip.",
    ),
    (
        "Playground", "🧪",
        "Generate synthetic manipulations from your own images to see how forgeries are actually built.",
        "Pure classical computer-vision, no ML model involved: seamless "
        "cloning for splicing, patch duplication for copy-move, Telea "
        "inpainting for content-aware fill, JPEG re-encoding for compression "
        "artifacts. Produces a ground-truth mask alongside each manipulation "
        "so you can see exactly where it happened.",
    ),
    (
        "Audio Detection", "🎙️",
        "Upload a voice clip and get a real/synthetic verdict for voice-clone and text-to-speech detection.",
        "Raw waveform resampled to 16kHz mono, zero-mean/unit-variance "
        "normalized, and fed through a wav2vec2-based classifier "
        "(ai8shiro/deepfake-audio-wav2vec2-ONNX) fine-tuned specifically for "
        "real-vs-synthetic speech -- a separate modality from the image/video "
        "models, with its own ONNX session and no shared preprocessing.",
    ),
    (
        "Quiz Mode", "🧠",
        "Test your own eye against the model -- guess real or fake, then see who's right.",
        "Generates a random challenge the same way Playground does, hides the "
        "ground truth until you guess, then reveals both your answer and the "
        "model's own prediction -- including its heatmap -- side by side.",
    ),
    (
        "Feedback", "📝",
        "Flag a wrong prediction so it can be folded back into future training.",
        "Submissions are logged to a structured CSV and can be merged into the "
        "training manifest for the next retraining pass -- a lightweight "
        "human-in-the-loop correction cycle, not an automatic live-retrain.",
    ),
]

PAGES_ORDER = [
    "Home", "Image Detection", "Video Detection", "Audio Detection", "Playground",
    "Quiz Mode", "Feedback", "About",
]
PAGE_ICONS = {
    "Home": "🏠", "Image Detection": "🖼️", "Video Detection": "🎞️", "Audio Detection": "🎙️",
    "Playground": "🧪", "Quiz Mode": "🧠", "Feedback": "📝", "About": "ℹ️",
}


# ---------------------------------------------------------------------------
# Styling -- only what config.toml can't express
# ---------------------------------------------------------------------------

def inject_accent_styles() -> None:
    st.html(
        """
<style>
.faillab-gradient-text {
    background: linear-gradient(90deg, #00f0ff, #8b5cf6 60%, #ff2fb0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.faillab-header { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.2rem; }
.faillab-header .icon { font-size: 2rem; }
.faillab-header h2 { font-size: 1.9rem; margin: 0; }
.faillab-subtitle {
    color: #9aa3c7;
    font-size: 1.02rem;
    margin: 0.1rem 0 1.4rem 0;
    border-left: 3px solid #8b5cf6;
    padding-left: 0.7rem;
}

/* Persistent top bar */
.app-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.4rem 0 0.9rem 0;
    border-bottom: 1px solid rgba(0, 240, 255, 0.18);
    margin-bottom: 1.3rem;
}
.app-topbar-brand { display: flex; align-items: center; gap: 0.5rem; font-family: 'Orbitron', sans-serif; font-size: 1.1rem; }
.app-topbar-crumb {
    font-size: 0.8rem;
    letter-spacing: 0.08em;
    color: #9aa3c7;
    border: 1px solid rgba(139, 92, 246, 0.35);
    border-radius: 999px;
    padding: 0.2rem 0.8rem;
}

/* Hero */
.hero-wrap { text-align: center; padding: 1rem 1rem 1rem 1rem; }
.hero-badge {
    display: inline-block;
    font-family: 'Orbitron', sans-serif;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    color: #00f0ff;
    border: 1px solid rgba(0, 240, 255, 0.4);
    border-radius: 999px;
    padding: 0.35rem 1rem;
    margin-bottom: 1.1rem;
    background: rgba(0, 240, 255, 0.06);
}
.hero-title {
    font-family: 'Orbitron', sans-serif;
    font-weight: 800;
    font-size: 3rem;
    letter-spacing: 0.03em;
    margin: 0;
    text-shadow: 0 0 60px rgba(139, 92, 246, 0.25);
}
.hero-motto {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.1rem;
    letter-spacing: 0.08em;
    color: #d7dbff;
    margin: 1rem 0 0.5rem 0;
}
.hero-sub { color: #9aa3c7; font-size: 1.02rem; max-width: 620px; margin: 0 auto; }

/* Home feature cards -- hover reveals the technical blurb */
.feat-icon {
    font-size: 2rem;
    display: inline-block;
    margin-bottom: 0.3rem;
    filter: drop-shadow(0 0 10px rgba(0, 240, 255, 0.35));
}
.feat-title { font-family: 'Orbitron', sans-serif; font-size: 1rem; color: #f1f3ff; margin-bottom: 0.3rem; }
.feat-desc { color: #a6addc; font-size: 0.9rem; line-height: 1.4; }
.feat-detail {
    color: #8891c4;
    font-size: 0.82rem;
    line-height: 1.45;
    max-height: 0;
    opacity: 0;
    overflow: hidden;
    transition: max-height 0.35s ease, opacity 0.35s ease, margin-top 0.35s ease;
}
div[class*="st-key-glass-feat-"]:hover .feat-detail {
    max-height: 260px;
    opacity: 1;
    margin-top: 0.6rem;
}
.faillab-pill {
    display: inline-block;
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    color: #00f0ff;
    border: 1px solid rgba(0, 240, 255, 0.35);
    border-radius: 999px;
    padding: 0.15rem 0.7rem;
    margin-bottom: 0.6rem;
}

/* Glass/3D lift on any container given a key starting with "glass-" */
div[class*="st-key-glass-"] {
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
}
div[class*="st-key-glass-"]:hover {
    transform: translateY(-4px);
    box-shadow: 0 16px 40px rgba(0, 240, 255, 0.18);
    border-color: rgba(0, 240, 255, 0.4) !important;
}

/* Sidebar nav buttons -- full-width, left-aligned, like real nav items */
section[data-testid="stSidebar"] div[class*="st-key-navbtn-"] button {
    justify-content: flex-start;
    text-align: left;
    font-family: 'Rajdhani', sans-serif;
    font-weight: 600;
    letter-spacing: 0.02em;
}

/* Scan-line loading animation (CSS keyframes keep animating client-side
   even while the Python script is blocked on a synchronous HTTP call). */
.scan-loader { padding: 0.9rem 0.2rem; }
.scan-bar {
    position: relative;
    height: 4px;
    border-radius: 4px;
    background: rgba(0, 240, 255, 0.08);
    overflow: hidden;
    margin-bottom: 0.55rem;
}
.scan-bar::after {
    content: "";
    position: absolute;
    top: 0; left: -40%;
    height: 100%; width: 40%;
    background: linear-gradient(90deg, transparent, #00f0ff, #8b5cf6, transparent);
    animation: scan-move 1.1s ease-in-out infinite;
}
@keyframes scan-move {
    0% { left: -40%; }
    100% { left: 100%; }
}
.scan-text {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.82rem;
    letter-spacing: 0.1em;
    color: #9aa3c7;
    animation: pulse-text 1.4s ease-in-out infinite;
}
@keyframes pulse-text {
    0%, 100% { opacity: 0.55; }
    50% { opacity: 1; }
}
</style>
"""
    )


@contextmanager
def loading_animation(message: str):
    """Drop-in replacement for `with st.spinner(...):` with a CSS scan-line
    animation instead of the default spinner -- the animation keeps running
    in the browser even while this synchronous `requests.post()` call blocks
    the Python script, since CSS keyframes are driven by the browser's
    renderer, not the Streamlit websocket connection."""
    placeholder = st.empty()
    placeholder.html(
        f"""
<div class="scan-loader">
    <div class="scan-bar"></div>
    <div class="scan-text">{message}</div>
</div>
"""
    )
    try:
        yield
    finally:
        placeholder.empty()


def page_header(icon: str, title: str, subtitle: str = "") -> None:
    st.html(
        f"""
<div class="faillab-header"><span class="icon">{icon}</span><h2 class="faillab-gradient-text">{title}</h2></div>
{f'<div class="faillab-subtitle">{subtitle}</div>' if subtitle else ""}
"""
    )


def go_to(page_name: str) -> None:
    st.session_state["current_page"] = page_name
    st.rerun()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def b64_to_image(b64_string: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_string)))


def image_to_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def render_predict_result(body: dict) -> None:
    cols = st.columns(3)
    with cols[0]:
        with st.container(border=True, key="glass-pred-label"):
            st.caption("Prediction")
            st.metric("Label", body["label"].upper(), delta=f"{body['confidence']:.0%} confidence")
            st.write(f"**Manipulation type:** {body['manipulation_type']}")
            st.write(body["explanation"])
    with cols[1]:
        with st.container(border=True, key="glass-pred-mask"):
            st.caption("Tampered-region mask")
            if body.get("mask"):
                st.image(b64_to_image(body["mask"]), use_container_width=True)
            else:
                st.info("No mask (image predicted real).")
    with cols[2]:
        with st.container(border=True, key="glass-pred-cam"):
            st.caption("Eigen-CAM heatmap")
            st.image(b64_to_image(body["gradcam"]), use_container_width=True)
            st.caption(
                "Heatmap quality depends on trained weights -- expect generic/noisy "
                "saliency until a trained model is exported (see models/README.md)."
            )

    second_opinion = body.get("second_opinion")
    if second_opinion:
        with st.container(border=True, key="glass-pred-second-opinion"):
            st.caption(f"Second opinion -- {second_opinion['model_name']}")
            agrees = (second_opinion["label"] == "ai_generated") == (body["label"] == "fake")
            st.metric(
                "Independent verdict",
                second_opinion["label"].replace("_", " ").upper(),
                delta=f"{second_opinion['confidence']:.0%} confidence",
                delta_color="normal" if agrees else "inverse",
            )
            st.write(second_opinion["explanation"])
            if not agrees:
                st.warning(
                    "This independent model disagrees with the primary verdict above -- "
                    "worth a closer manual look."
                )


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_home() -> None:
    st.html(
        f"""
<div class="hero-wrap">
    <div class="hero-badge">⚡ AI-POWERED FORENSICS</div>
    <div class="hero-title faillab-gradient-text">{settings.app.name.upper()}</div>
    <div class="hero-motto">"{MOTTO}"</div>
    <p class="hero-sub">{TAGLINE}</p>
</div>
"""
    )

    st.html('<div class="faillab-pill">GUIDE -- hover a card for how it works, click to open it</div>')

    cols = st.columns(3)
    for i, (page_name, icon, short_desc, detail) in enumerate(FEATURE_GUIDE):
        with cols[i % 3]:
            with st.container(border=True, key=f"glass-feat-{i}"):
                st.html(
                    f"""
<div class="feat-icon">{icon}</div>
<div class="feat-title">{page_name}</div>
<div class="feat-desc">{short_desc}</div>
<div class="feat-detail">{detail}</div>
"""
                )
                if st.button(f"Open {page_name} →", key=f"navbtn-home-{i}", use_container_width=True):
                    go_to(page_name)

    with st.container(border=True, key="glass-how-it-works"):
        st.html(
            """
<div class="feat-title">⚙️ How it works, briefly</div>
<div class="feat-desc">
Every image is checked for real/fake, the specific manipulation type,
and (for fakes) where in the image it happened. Detection runs on
ONNX Runtime -- no PyTorch needed to serve predictions -- and
explanations come from Eigen-CAM, a gradient-free saliency method
that works directly off the exported model's own feature map.
Pick a page from the sidebar, or click a card above, to get started.
</div>
"""
        )


def page_image_detection() -> None:
    page_header("🖼️", "Image Detection", "Upload a photo for a real/fake verdict, tampered-region mask, and explainability heatmap.")
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded is None:
        return

    st.image(uploaded, caption="Uploaded image", width=300)
    if st.button("Analyze image", type="primary"):
        with loading_animation("ANALYZING IMAGE..."):
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
    page_header("🎞️", "Video Detection", "Frames are sampled, scored individually, and combined into one overall verdict.")
    uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded is None:
        return

    st.video(uploaded)
    if st.button("Analyze video", type="primary"):
        with loading_animation("EXTRACTING & ANALYZING FRAMES..."):
            response = requests.post(
                f"{API_BASE_URL}/predict/video",
                files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                timeout=300,
            )
        if response.status_code != 200:
            st.error(f"Request failed: {response.status_code} {response.text}")
            return

        body = response.json()
        with st.container(border=True, key="glass-video-verdict"):
            st.metric("Verdict", body["label"].upper(), delta=f"{body['confidence']:.0%} confidence")
            st.write(
                f"{body['fake_frames']} of {body['total_frames_sampled']} sampled frames "
                f"flagged fake ({body['fake_frame_ratio']:.0%})."
            )
            st.write(body["explanation"])
        st.dataframe(body["frame_results"], use_container_width=True)


def page_audio_detection() -> None:
    page_header(
        "🎙️", "Audio Detection",
        "Upload a voice clip for a real/synthetic verdict -- catches voice cloning and text-to-speech generation.",
    )
    uploaded = st.file_uploader("Upload an audio clip", type=["wav", "flac", "ogg"])
    if uploaded is None:
        return

    st.audio(uploaded)
    if st.button("Analyze audio", type="primary"):
        with loading_animation("ANALYZING AUDIO..."):
            response = requests.post(
                f"{AUDIO_API_BASE_URL}/predict/audio",
                files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                timeout=60,
            )
        if response.status_code != 200:
            st.error(f"Request failed: {response.status_code} {response.text}")
            return

        body = response.json()
        with st.container(border=True, key="glass-audio-verdict"):
            st.caption("Prediction")
            st.metric("Label", body["label"].upper(), delta=f"{body['confidence']:.0%} confidence")
            st.write(body["explanation"])


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
    page_header("🧪", "Playground", "Generate synthetic manipulations from your own images to see how forgeries are built.")
    uploaded = st.file_uploader("Upload a source image", type=["jpg", "jpeg", "png", "bmp"], key="playground_upload")
    manipulation_type = st.selectbox("Manipulation type", MANIPULATION_TYPES)

    if uploaded is None:
        return

    st.image(uploaded, caption="Source image", width=300)
    if st.button("Generate", type="primary"):
        with loading_animation(f"GENERATING {manipulation_type.upper()}..."):
            body = _generate_playground_image(manipulation_type, uploaded.getvalue(), uploaded.name)
        if body is None:
            return

        cols = st.columns(2)
        with cols[0]:
            with st.container(border=True, key="glass-pg-image"):
                st.caption("Manipulated image")
                st.image(b64_to_image(body["image"]), use_container_width=True)
        with cols[1]:
            with st.container(border=True, key="glass-pg-mask"):
                st.caption("Ground-truth mask")
                if body.get("mask"):
                    st.image(b64_to_image(body["mask"]), use_container_width=True)
                else:
                    st.info("This manipulation type doesn't produce a localized mask.")


def page_quiz() -> None:
    page_header("🧠", "Quiz Mode", "Test your own eye against the model -- guess real or fake, then see who's right.")

    uploaded = st.file_uploader("Upload a source image", type=["jpg", "jpeg", "png", "bmp"], key="quiz_upload")
    if uploaded is None:
        return

    if st.button("New challenge", type="primary") or "quiz_challenge" not in st.session_state:
        challenge_type = random.choice(MANIPULATION_TYPES + ["none"])
        with loading_animation("PREPARING CHALLENGE..."):
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

        with loading_animation("RUNNING AI PREDICTION..."):
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
    page_header("📝", "Feedback", "Flag a wrong prediction so it can be folded back into future training.")

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

    if st.button("Submit feedback", type="primary"):
        with loading_animation("SUBMITTING..."):
            response = requests.post(f"{API_BASE_URL}/feedback", json=payload, timeout=30)
        if response.status_code == 200:
            st.success(f"Thanks! Feedback ID: {response.json()['feedback_id']}")
        else:
            st.error(f"Submission failed: {response.status_code} {response.text}")


def page_about() -> None:
    page_header("ℹ️", "About", "Architecture, training roadmap, and resource-constraint notes.")
    with st.container(border=True, key="glass-about"):
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


PAGES = {
    "Home": page_home,
    "Image Detection": page_image_detection,
    "Video Detection": page_video_detection,
    "Audio Detection": page_audio_detection,
    "Playground": page_playground,
    "Quiz Mode": page_quiz,
    "Feedback": page_feedback,
    "About": page_about,
}


def render_sidebar_nav() -> None:
    st.sidebar.html(
        f"""
<div style="text-align:center; padding: 0.5rem 0 1.2rem 0;">
    <div class="faillab-gradient-text" style="font-family:'Orbitron',sans-serif; font-size:1.3rem;">
        {settings.app.name.upper()}
    </div>
    <div style="color:#8089b3; font-size:0.72rem; letter-spacing:0.08em;">{MOTTO}</div>
</div>
"""
    )
    current = st.session_state["current_page"]
    for name in PAGES_ORDER:
        is_active = name == current
        if st.sidebar.button(
            f"{PAGE_ICONS[name]}  {name}",
            key=f"navbtn-sidebar-{name}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            if not is_active:
                go_to(name)


def main() -> None:
    st.set_page_config(page_title=settings.app.name, layout="wide", page_icon="🛰️")
    st.session_state.setdefault("current_page", "Home")
    inject_accent_styles()
    render_sidebar_nav()

    current = st.session_state["current_page"]
    st.html(
        f"""
<div class="app-topbar">
    <div class="app-topbar-brand">
        <span>🛰️</span>
        <span class="faillab-gradient-text">{settings.app.name.upper()}</span>
    </div>
    <div class="app-topbar-crumb">{PAGE_ICONS[current]} {current}</div>
</div>
"""
    )
    PAGES[current]()


if __name__ == "__main__":
    main()
