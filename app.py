"""Consent-gated MediaPipe + optional SAM 2.1 hand gesture lab."""
import time
from pathlib import Path

import joblib
import streamlit as st
from gesture_core import LABELS, analyze, create_recognizer, features, overlay
from PIL import Image, ImageOps

st.set_page_config(
    page_title="Hand Gesture Lab · MediaPipe + SAM 2.1", page_icon="✋", layout="wide"
)
st.title("✋ Hand Gesture Lab")
st.caption(
    "Peace · Stop · Thumbs Up | MediaPipe recognition + optional SAM 2.1 hand segmentation"
)


@st.cache_resource
def get_engine():
    return create_recognizer()


@st.cache_resource
def get_custom():
    path = Path(__file__).parent / "models" / "landmark_classifier.joblib"
    return joblib.load(path) if path.exists() else None


@st.cache_resource
def get_sam(device):
    from sam2_core import load_predictor

    return load_predictor(device)


st.sidebar.header("Model settings")
custom_available = (
    Path(__file__).parent / "models" / "landmark_classifier.joblib"
).exists()
mode = st.sidebar.radio(
    "Classifier",
    ["Pretrained MediaPipe", "Custom landmark model"]
    if custom_available
    else ["Pretrained MediaPipe"],
)
threshold = st.sidebar.slider("Minimum gesture score", 0.0, 1.0, 0.55, 0.05)
use_sam = st.sidebar.checkbox(
    "Enable SAM 2.1 segmentation",
    value=False,
    help="Optional; requires installing SAM 2 and downloading its checkpoint",
)
device = (
    st.sidebar.selectbox("SAM 2.1 device", ["cpu", "mps", "cuda"], index=0)
    if use_sam
    else "cpu"
)
st.sidebar.info(
    "Model scores are not calibrated probabilities. Segmentation masks are visual explanations, not proof of correct recognition."
)

st.subheader("Privacy and consent")
st.info(
    "This app processes images in the Python process running Streamlit. It does not intentionally send photos to external inference APIs or save uploads to a dataset. If hosted remotely, uploaded images travel to that Streamlit server. Streamlit session and server logs/caches may still exist. Model downloads require network access."
)
consent = st.checkbox(
    "I own this image or have permission from people shown to process it for this educational demo, and I understand the processing notice above.",
    value=False,
)
if not consent:
    st.warning("Confirm consent above to enable image upload or camera capture.")
    st.stop()

upload = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png", "webp"])
shot = st.camera_input("Or capture a hand gesture")
source = shot if shot is not None else upload
if source is None:
    st.info("Upload a photo or take a camera picture to begin.")
    st.stop()

try:
    image = ImageOps.exif_transpose(Image.open(source)).convert("RGB")
    start = time.perf_counter()
    image, landmarks, scores, label, _ = analyze(get_engine(), image)
    if mode == "Custom landmark model" and landmarks:
        model = get_custom()
        f = features(landmarks)
        if model is not None and f is not None:
            p = model.predict_proba([f])[0]
            scores = {k: float(v) for k, v in zip(model.classes_, p)}
            label = max(scores, key=scores.get)
    classification_ms = (time.perf_counter() - start) * 1000
    col1, col2 = st.columns(2)
    col1.image(image, caption="Input", use_container_width=True)
    col2.image(
        overlay(image, landmarks),
        caption="MediaPipe 21-point landmark overlay",
        use_container_width=True,
    )
    if landmarks is None:
        st.error("No hand detected. Improve lighting and show the whole hand.")
    elif label is None or scores.get(label, 0) < threshold:
        st.warning("Uncertain or unsupported gesture.")
    else:
        st.success(
            f'Prediction: {label.replace("_"," ").title()} · score {scores[label]:.1%}'
        )
    for k in LABELS:
        st.progress(
            min(1.0, max(0.0, scores.get(k, 0.0))),
            text=f'{k.replace("_"," ").title()}: {scores.get(k,0):.1%}',
        )
    st.caption(f"MediaPipe + classifier: {classification_ms:.1f} ms for this image")

    if use_sam:
        st.subheader("SAM 2.1 — prompt-guided hand segmentation")
        if landmarks is None:
            st.info("SAM 2.1 requires MediaPipe to detect a hand first.")
        else:
            from sam2_core import segment_hand

            sam_start = time.perf_counter()
            seg = segment_hand(get_sam(device), image, landmarks)
            sam_ms = (time.perf_counter() - sam_start) * 1000
            a, b, c = st.columns(3)
            a.image(
                seg["overlay"], caption="SAM 2.1 hand overlay", use_container_width=True
            )
            b.image(
                seg["mask"], caption="SAM 2.1 binary mask", use_container_width=True
            )
            c.image(
                seg["cutout"], caption="Background removed", use_container_width=True
            )
            st.caption(
                f'SAM 2.1: {sam_ms:.1f} ms · SAM mask score: {seg["sam_score"]:.3f} · Foreground area: {seg["mask_fraction"]:.1%}'
            )
            st.caption(
                "Mask is generated from a MediaPipe landmark-derived box prompt. This is segmentation, not a second gesture classifier."
            )
            if st.checkbox(
                "Run recognition again on the SAM-masked image (ablation preview)"
            ):
                start2 = time.perf_counter()
                _, masked_lm, masked_scores, masked_label, _ = analyze(
                    get_engine(), seg["cutout"]
                )
                if mode == "Custom landmark model":
                    masked_f = features(masked_lm)
                    if masked_f is not None and get_custom() is not None:
                        masked_p = get_custom().predict_proba([masked_f])[0]
                        masked_scores = dict(
                            zip(get_custom().classes_, map(float, masked_p))
                        )
                        masked_label = max(masked_scores, key=masked_scores.get)
                    else:
                        masked_scores = {k: 0.0 for k in LABELS}
                        masked_label = None
                st.write(
                    {
                        "Original prediction": label,
                        "Original score": round(scores.get(label, 0), 3)
                        if label
                        else None,
                        "Masked prediction": masked_label,
                        "Masked score": round(masked_scores.get(masked_label, 0), 3)
                        if masked_label
                        else None,
                        "Masked inference ms": round(
                            (time.perf_counter() - start2) * 1000, 1
                        ),
                    }
                )
                st.caption(
                    "A single example does not establish an accuracy improvement. Use evaluate_sam_ablation.py on the test set."
                )
except Exception as exc:
    st.error(f"Processing failed: {exc}")
    st.caption(
        "If SAM 2.1 is enabled, check the separate SAM installation and checkpoint instructions in README.md."
    )