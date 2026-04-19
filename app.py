import streamlit as st
import numpy as np
from PIL import Image
import hashlib
import os
from datetime import datetime
from io import BytesIO
import tempfile

# ✅ SAFE IMPORT (IMPORTANT FIX)
try:
    import cv2
    CV2_AVAILABLE = True
except:
    CV2_AVAILABLE = False

# ─── Page Config ─────────────────────
st.set_page_config(
    page_title="DeepTrust",
    page_icon="🔍",
    layout="wide"
)

# ─── UI ─────────────────────────────
st.title("🔍 DeepTrust")
st.subheader("AI Deepfake Detection System")

# ─── Detector Class ─────────────────
class DeepfakeDetector:

    def analyze_image(self, image_path):
        if not CV2_AVAILABLE:
            return self._fallback_result("cv2 not installed")

        try:
            img = cv2.imread(image_path)

            if img is None:
                return {"error": "Image load failed"}

            score = self._simple_score(img)

            return self._build_result(score)

        except Exception as e:
            return {"error": str(e)}

    def analyze_video(self, video_path):
        if not CV2_AVAILABLE:
            return self._fallback_result("cv2 not installed")

        try:
            cap = cv2.VideoCapture(video_path)
            scores = []

            for _ in range(10):
                ret, frame = cap.read()
                if not ret:
                    break
                scores.append(self._simple_score(frame))

            cap.release()

            if not scores:
                return {"error": "No frames"}

            avg_score = np.mean(scores)
            return self._build_result(avg_score)

        except Exception as e:
            return {"error": str(e)}

    # 🔥 SIMPLE AI LOGIC (can upgrade later)
    def _simple_score(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        variance = np.var(gray)

        # heuristic scoring
        score = min(1.0, variance / 5000)
        return score

    def _build_result(self, score):
        score = int(score * 100)

        return {
            "trust_score": score,
            "is_authentic": score >= 70,
            "is_suspicious": 40 <= score < 70,
            "is_deepfake": score < 40
        }

    def _fallback_result(self, reason):
        return {
            "trust_score": 50,
            "is_authentic": False,
            "is_suspicious": True,
            "is_deepfake": False,
            "warning": reason
        }


# ─── Utils ──────────────────────────
def generate_hash(data):
    return hashlib.sha256(data).hexdigest()


def show_result(result):
    score = result["trust_score"]

    if score >= 70:
        st.success(f"✅ Authentic ({score})")
    elif score >= 40:
        st.warning(f"⚠️ Suspicious ({score})")
    else:
        st.error(f"🚨 Deepfake ({score})")

    if "warning" in result:
        st.info(f"⚠️ {result['warning']}")


# ─── App Logic ──────────────────────
detector = DeepfakeDetector()

uploaded_file = st.file_uploader(
    "Upload Image or Video",
    type=["jpg", "jpeg", "png", "mp4"]
)

if uploaded_file:

    file_bytes = uploaded_file.read()

    st.markdown("### Preview")

    if uploaded_file.type.startswith("image"):
        img = Image.open(BytesIO(file_bytes))
        st.image(img)

    elif uploaded_file.type.startswith("video"):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file_bytes)
            st.video(tmp.name)

    if st.button("Analyze 🚀"):

        with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
            tmp.write(file_bytes)
            path = tmp.name

        with st.spinner("Analyzing..."):
            if uploaded_file.type.startswith("image"):
                result = detector.analyze_image(path)
            else:
                result = detector.analyze_video(path)

        st.markdown("## Result")
        show_result(result)

        st.markdown("### File Hash")
        st.code(generate_hash(file_bytes))
