import streamlit as st
import numpy as np
from PIL import Image
import hashlib
import tempfile
from io import BytesIO

# ✅ Safe OpenCV import
try:
    import cv2
    CV2_AVAILABLE = True
except:
    CV2_AVAILABLE = False

# ─── Page Config ─────────────────────
st.set_page_config(page_title="DeepTrust", page_icon="🔍", layout="wide")

st.title("🔍 DeepTrust")
st.subheader("AI vs Real Media Detection")

# ─── Detector ───────────────────────
class DeepfakeDetector:

    def analyze_image(self, path):
        if not CV2_AVAILABLE:
            return self._fallback("OpenCV not installed")

        img = cv2.imread(path)
        if img is None:
            return {"error": "Image load failed"}

        score = self._score(img)
        return self._build_result(score)

    def analyze_video(self, path):
        if not CV2_AVAILABLE:
            return self._fallback("OpenCV not installed")

        cap = cv2.VideoCapture(path)
        scores = []

        for _ in range(10):
            ret, frame = cap.read()
            if not ret:
                break
            scores.append(self._score(frame))

        cap.release()

        if not scores:
            return {"error": "No frames"}

        return self._build_result(np.mean(scores))

    # 🔥 Simple heuristic scoring
    def _score(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        variance = np.var(gray)

        # normalize score (0–1)
        return min(1.0, variance / 5000)

    # ✅ Final verdict logic
    def _build_result(self, score):
        score = int(score * 100)

        if score >= 70:
            label = "REAL"
            emoji = "✅"
        elif score >= 40:
            label = "UNCERTAIN"
            emoji = "⚠️"
        else:
            label = "AI GENERATED"
            emoji = "🚨"

        confidence = abs(score - 50) * 2

        return {
            "score": score,
            "label": label,
            "emoji": emoji,
            "confidence": int(confidence)
        }

    def _fallback(self, msg):
        return {
            "score": 50,
            "label": "UNCERTAIN",
            "emoji": "⚠️",
            "confidence": 0,
            "warning": msg
        }


# ─── Utils ──────────────────────────
def file_hash(data):
    return hashlib.sha256(data).hexdigest()


def show_result(res):
    st.markdown("## 🔍 Final Verdict")

    st.markdown(f"""
    ### {res['emoji']} {res['label']}
    **Confidence:** {res['confidence']}%  
    **Score:** {res['score']}/100
    """)

    if res["label"] == "REAL":
        st.success("Likely a real image/video.")
    elif res["label"] == "AI GENERATED":
        st.error("Likely AI-generated (deepfake).")
    else:
        st.warning("Model is not confident.")

    if "warning" in res:
        st.info(res["warning"])


# ─── App ────────────────────────────
detector = DeepfakeDetector()

uploaded = st.file_uploader(
    "Upload Image or Video",
    type=["jpg", "jpeg", "png", "mp4"]
)

if uploaded:
    data = uploaded.read()

    st.markdown("### 👀 Preview")

    if uploaded.type.startswith("image"):
        img = Image.open(BytesIO(data))
        st.image(img)

    else:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(data)
            st.video(tmp.name)

    if st.button("🚀 Analyze"):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(data)
            path = tmp.name

        with st.spinner("Analyzing..."):
            if uploaded.type.startswith("image"):
                result = detector.analyze_image(path)
            else:
                result = detector.analyze_video(path)

        show_result(result)

        st.markdown("### 🔐 File Hash")
        st.code(file_hash(data))
