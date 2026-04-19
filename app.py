import streamlit as st
import numpy as np
from PIL import Image
import tempfile
from io import BytesIO
import hashlib

# 🔥 REAL AI MODEL (Transformers)
from transformers import pipeline

# Load model once
@st.cache_resource
def load_model():
    return pipeline(
        "image-classification",
        model="nateraw/deepfake-detection-model"  # lightweight demo model
    )

detector_model = load_model()

# ─── UI ─────────────────────────────
st.set_page_config(page_title="DeepTrust AI", page_icon="🔍", layout="wide")

st.title("🔍 DeepTrust (Real AI Model)")
st.subheader("Deepfake Detection using AI Model")

# ─── Utils ──────────────────────────
def file_hash(data):
    return hashlib.sha256(data).hexdigest()


def analyze_image(image):
    results = detector_model(image)

    # Example output:
    # [{'label': 'FAKE', 'score': 0.92}]

    label = results[0]["label"].upper()
    confidence = int(results[0]["score"] * 100)

    return label, confidence


def analyze_video(path):
    import cv2

    cap = cv2.VideoCapture(path)
    preds = []

    for _ in range(8):  # sample frames
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)

        result = detector_model(img)[0]
        preds.append(result)

    cap.release()

    if not preds:
        return "UNCERTAIN", 0

    # average prediction
    fake_scores = [
        p["score"] for p in preds if "fake" in p["label"].lower()
    ]

    real_scores = [
        p["score"] for p in preds if "real" in p["label"].lower()
    ]

    avg_fake = np.mean(fake_scores) if fake_scores else 0
    avg_real = np.mean(real_scores) if real_scores else 0

    if avg_fake > avg_real:
        return "AI GENERATED", int(avg_fake * 100)
    else:
        return "REAL", int(avg_real * 100)


def show_result(label, confidence):
    st.markdown("## 🔍 Final Verdict")

    if "REAL" in label:
        st.success(f"✅ REAL ({confidence}%)")
    elif "FAKE" in label or "AI" in label:
        st.error(f"🚨 AI GENERATED ({confidence}%)")
    else:
        st.warning(f"⚠️ UNCERTAIN ({confidence}%)")


# ─── App ────────────────────────────
uploaded = st.file_uploader(
    "Upload Image or Video",
    type=["jpg", "jpeg", "png", "mp4"]
)

if uploaded:
    data = uploaded.read()

    st.markdown("### 👀 Preview")

    if uploaded.type.startswith("image"):
        img = Image.open(BytesIO(data)).convert("RGB")
        st.image(img)

    else:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(data)
            st.video(tmp.name)

    if st.button("🚀 Analyze with AI"):
        with st.spinner("Running AI model..."):

            if uploaded.type.startswith("image"):
                img = Image.open(BytesIO(data)).convert("RGB")
                label, confidence = analyze_image(img)

            else:
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp.write(data)
                    path = tmp.name

                label, confidence = analyze_video(path)

        show_result(label, confidence)

        st.markdown("### 🔐 File Hash")
        st.code(file_hash(data))
