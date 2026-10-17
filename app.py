import hashlib
from datetime import datetime, timezone
from io import BytesIO

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from care import DISASTERS, analyze_scene, format_prediction, predict_condition
from detector import AnimalDetector
from emailer import is_configured, send_rescue_email, smtp_config
from fusion import fuse_detections
from report import format_analysis_report
from thermal import create_simulated_thermal, process_thermal_frame
from triage import HEALTH_OBSERVATIONS, build_health_screen

st.set_page_config(page_title="FUR-REAL | Animal Rescue", page_icon="🐾", layout="wide")

NOTICE = ("Research prototype. Predictions are estimates from the photo and your checklist - "
          "not a veterinary diagnosis. Always confirm in person and contact professionals.")


@st.cache_resource(show_spinner="Loading animal detection model (first time only)...")
def load_detector(model_path):
    return AnimalDetector(model_path)


def read_image_bytes(data, description):
    try:
        with Image.open(BytesIO(data)) as image:
            img = ImageOps.exif_transpose(image).convert("RGB")
            return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    except (UnidentifiedImageError, OSError, ValueError) as error:
        st.error(f"Could not read the {description} image: {error}")
        return None


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def png(img):
    ok, enc = cv2.imencode(".png", img)
    return enc.tobytes() if ok else b""


def health_screen_now():
    selected = {k for k, _ in HEALTH_OBSERVATIONS if st.session_state.get(f"health_{k}")}
    if not selected and not st.session_state.get("health_touched"):
        return {"status": "not_assessed", "observations": [], "source": "not_assessed"}
    return build_health_screen(selected)


st.title("🐾 FUR-REAL — Animal Search & Rescue")
st.caption(NOTICE)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Setup")
    input_mode = st.radio("Image source", ["Upload image", "Take photo (camera)", "Live webcam detection"])
    model_path = st.text_input("YOLO model file", value="yolo26n.pt").strip() or "yolo26n.pt"
    confidence = st.slider("Detection confidence", 0.10, 0.90, 0.25, 0.05)
    disaster_choice = st.selectbox("Situation / disaster", DISASTERS)
    location = st.text_input("Location (address / GPS)", "")
    st.divider()
    thermal_upload = st.file_uploader("Thermal image (optional)",
                                      type=["bmp", "jpeg", "jpg", "png", "tif", "tiff", "webp"])
    hotspot_percentile = st.slider("Hotspot percentile", 70, 99, 90)
    min_area = st.number_input("Min hotspot area (px)", 1, 100000, 20, 5)
    show_sim = st.checkbox("Show thermal-style visualization", False)

# ---------------- Input ----------------
visible_image, source = None, ""
if input_mode == "Upload image":
    up = st.file_uploader("Upload a photo of the animal / scene",
                          type=["bmp", "jpeg", "jpg", "png", "tif", "tiff", "webp"])
    if up is not None:
        visible_image, source = read_image_bytes(up.getvalue(), "uploaded"), f"Uploaded: {up.name}"
elif input_mode == "Take photo (camera)":
    shot = st.camera_input("Take a photo")
    if shot is not None:
        visible_image, source = read_image_bytes(shot.getvalue(), "camera"), "Camera photo"
else:
    from streamlit_webrtc import webrtc_streamer
    from live_video import LiveAnimalVideoProcessor
    ctx = webrtc_streamer(
        key=f"live-{model_path}-{confidence:.2f}",
        video_processor_factory=lambda: LiveAnimalVideoProcessor(model_path, confidence),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )
    if ctx.video_processor is not None and st.button("📸 Save current frame for analysis"):
        frame = ctx.video_processor.get_latest_frame()
        if frame is None:
            st.warning("Waiting for the first webcam frame...")
        else:
            st.session_state["live_frame"] = frame
    visible_image = st.session_state.get("live_frame")
    source = "Live webcam frame"

thermal_image = read_image_bytes(thermal_upload.getvalue(), "thermal") if thermal_upload else None

if visible_image is None:
    st.info("⬆️ Upload or capture a photo to start. Analysis runs automatically.")
    st.stop()

# ---------------- Health checklist ----------------
with st.expander("🩺 What do you see? (handler checklist - improves the prediction)", expanded=False):
    cols = st.columns(2)
    for i, (key, label) in enumerate(HEALTH_OBSERVATIONS):
        cols[i % 2].checkbox(label, key=f"health_{key}",
                             on_change=lambda: st.session_state.update(health_touched=True))
health_screen = health_screen_now()

same_coords = False
if thermal_image is not None:
    same_coords = st.checkbox("Visible and thermal images are registered to the same coordinates")

# ---------------- Analysis (automatic, cached by image signature) ----------------
sig = hashlib.sha256(visible_image.tobytes()).hexdigest() + f"{model_path}{confidence}"
if thermal_image is not None:
    sig += hashlib.sha256(thermal_image.tobytes()).hexdigest() + f"{hotspot_percentile}{min_area}{same_coords}"

if st.session_state.get("sig") != sig:
    try:
        with st.spinner("Analyzing image..."):
            detector = load_detector(model_path)
            annotated, animals = detector.detect(visible_image, confidence=confidence)
            scene = analyze_scene(visible_image)
            hotspots, thermal_annot, fusion = [], None, []
            if thermal_image is not None:
                thermal_annot, _, hotspots = process_thermal_frame(
                    thermal_image, percentile=hotspot_percentile, minimum_area=int(min_area))
                if same_coords and visible_image.shape[:2] == thermal_image.shape[:2]:
                    fusion = fuse_detections(animals, hotspots)
        st.session_state.update(sig=sig, res={
            "annotated": annotated, "animals": animals, "scene": scene, "hotspots": hotspots,
            "thermal_annot": thermal_annot, "fusion": fusion, "model": detector.model_path,
        })
    except Exception as error:  # noqa: BLE001
        st.error(f"Analysis failed: {error}")
        st.info("Fix: run `pip install -U ultralytics` (yolo26n.pt needs ultralytics 8.4+). "
                "The app also auto-falls back to yolo11n.pt / yolov8n.pt if internet is available.")
        st.stop()

r = st.session_state["res"]
prediction = predict_condition(r["animals"], r["scene"], disaster_choice, health_screen)

# ---------------- Results ----------------
c1, c2 = st.columns(2)
with c1:
    st.subheader("Detected animals")
    st.image(rgb(r["annotated"]), width="stretch")
with c2:
    if r["thermal_annot"] is not None:
        st.subheader("Thermal hotspots")
        st.image(rgb(r["thermal_annot"]), width="stretch")
    elif show_sim:
        st.subheader("Thermal-style view (simulated)")
        st.image(rgb(create_simulated_thermal(visible_image)), width="stretch")
    else:
        st.subheader("Scene cues")
        for cue in r["scene"]["cues"] or ["No strong hazard cues detected in the photo"]:
            st.write("• " + cue)

m = st.columns(4)
m[0].metric("Animals found", len(r["animals"]))
m[1].metric("Species", ", ".join(dict.fromkeys(a["label"] for a in r["animals"])) or "—")
m[2].metric("Situation", prediction["disaster"])
m[3].metric("Risk", f"{prediction['risk_level']} ({prediction['risk_score']}/5)")

if not r["animals"]:
    st.warning("No supported animal detected. Try a clearer/closer photo or lower the confidence slider. "
               "Guidance below uses general advice.")
for i, a in enumerate(r["animals"], 1):
    st.write(f"**{i}. {a['label'].title()}** — {a['confidence']:.0%} confidence")

level_box = {"CRITICAL": st.error, "HIGH": st.error, "MODERATE": st.warning}.get(prediction["risk_level"], st.info)
level_box(f"**Prediction:** likely *{prediction['disaster']}* situation. Possible conditions: "
          + ", ".join(prediction["likely_conditions"]))

t1, t2 = st.tabs(["🚑 What to do now", "💊 Treatment guide by animal"])
with t1:
    for i, step in enumerate(prediction["before_rescue"], 1):
        st.write(f"{i}. {step}")
with t2:
    for s in prediction["species"]:
        st.markdown(f"### {s['label'].title()}")
        st.write(f"**Handling:** {s['handling']}")
        st.write("**Recommended treatments:**")
        for t in s["treatments"]:
            st.write("• " + t)
        st.write(f"⚠️ **Avoid:** {s['avoid']}")

# ---------------- Report ----------------
h, w = r["annotated"].shape[:2]
report = format_analysis_report(
    created_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    image_shape=[h, w], model=r["model"], confidence_threshold=confidence, animals=r["animals"],
    thermal_shape=None if thermal_image is None else thermal_image.shape[:2], hotspots=r["hotspots"],
    hotspot_percentile=hotspot_percentile, fusion_enabled=bool(r["fusion"]), fusion_results=r["fusion"],
    health_screen=health_screen, source=source,
) + format_prediction(prediction) + (f"\nLOCATION: {location}\n" if location else "")

with st.expander("📄 Full report"):
    st.text(report)
d1, d2 = st.columns(2)
d1.download_button("Download report (.txt)", report.encode(), "fur-real-report.txt", "text/plain")
d2.download_button("Download annotated image", png(r["annotated"]), "fur-real-detections.png", "image/png")

# ---------------- Email (direct SMTP send) ----------------
st.subheader("📧 Send alert to rescue team")
cfg = smtp_config()
to = st.text_input("Rescue team email", value=cfg["rescue_to"])
note = st.text_area("Extra message (optional)", "")
if not is_configured(cfg):
    st.warning("Email not configured yet: fill SMTP_USER and SMTP_PASSWORD in `.streamlit/secrets.toml`, "
               "then restart the app. Emails are then sent directly - no Outlook/mail app needed.")
if st.button("🚨 Send to rescue team now", type="primary", disabled=not is_configured(cfg)):
    if not to or "@" not in to:
        st.error("Enter a valid rescue team email.")
    else:
        species = ", ".join(dict.fromkeys(a["label"] for a in r["animals"])) or "unknown animal"
        subject = f"[FUR-REAL {prediction['risk_level']}] {species} - {prediction['disaster']}"
        body = (f"Rescue alert from FUR-REAL\nLocation: {location or 'not provided'}\n"
                f"{('Note: ' + note) if note else ''}\n\n{report}")
        try:
            with st.spinner("Sending..."):
                send_rescue_email(to, subject, body, attachments=[
                    ("animal-detection.png", png(r["annotated"]), "image", "png"),
                    ("fur-real-report.txt", report.encode(), "text", "plain"),
                ], cfg=cfg)
            st.success(f"✅ Alert sent to {to}")
        except Exception as error:  # noqa: BLE001
            st.error(f"Email failed: {error}. Check your app password and SMTP settings.")

st.caption("FUR-REAL | Research prototype — not a certified rescue or veterinary instrument")
