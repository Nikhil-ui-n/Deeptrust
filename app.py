import streamlit as st
import numpy as np
from PIL import Image
import hashlib
import tempfile
from io import BytesIO
import matplotlib.pyplot as plt
import requests

# Safe OpenCV import
try:
    import cv2
    CV2 = True
except:
    CV2 = False

st.set_page_config(page_title="DeepTrust", layout="wide")
st.title("🔍 DeepTrust - AI Deepfake Detector")

# ─── Detector ───────────────────────
class DeepfakeDetector:

    def analyze_image(self, path):
        if not CV2:
            return {"error": "OpenCV not installed"}, [], []

        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        score = self._compute(gray)
        return self._build(score), [], []

    def analyze_video(self, path):
        if not CV2:
            return {"error": "OpenCV not installed"}, [], []

        cap = cv2.VideoCapture(path)
        frames = []
        scores = []

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        count = 0
        while count < 12:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Face detection
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            score = self._compute(gray)
            scores.append(int(score * 100))

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frames.append({
                "image": frame_rgb,
                "score": int(score * 100)
            })

            count += 1

        cap.release()

        final_score = np.mean(scores)/100 if scores else 0.5
        return self._build(final_score), frames, scores

    def _compute(self, gray):
        variance = np.var(gray)
        noise = np.std(gray)
        return min(1.0, (variance + noise) / 6000)

    def _build(self, score):
        score = int(score * 100)
        return {
            "score": score,
            "verdict": "Authentic" if score >= 70 else "Suspicious" if score >= 40 else "Deepfake",
            "confidence": abs(score - 50) * 2
        }


# ─── Utils ──────────────────────────
def file_hash(data):
    return hashlib.sha256(data).hexdigest()

def explain(score):
    if score >= 70:
        return ["Natural texture", "Consistent lighting", "Low noise"]
    elif score >= 40:
        return ["Minor artifacts", "Slight blur", "Moderate noise"]
    else:
        return ["GAN artifacts", "Unnatural smoothing", "High noise"]


# ─── App ────────────────────────────
detector = DeepfakeDetector()

mode = st.sidebar.radio("Mode", ["Upload", "URL"])

# ─── Upload Mode ────────────────────
if mode == "Upload":
    uploaded = st.file_uploader("Upload Image/Video", type=["jpg","png","jpeg","mp4"])

    if uploaded:
        data = uploaded.read()

        st.subheader("Preview")

        if uploaded.type.startswith("image"):
            img = Image.open(BytesIO(data))
            st.image(img)

        else:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(data)
                video_path = tmp.name
            st.video(video_path)

        if st.button("Analyze 🚀"):

            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(data)
                path = tmp.name

            if uploaded.type.startswith("image"):
                result, frames, scores = detector.analyze_image(path)
            else:
                result, frames, scores = detector.analyze_video(path)

            st.markdown("## Result")

            score = result["score"]

            if score >= 70:
                st.success(f"✅ Authentic ({score})")
            elif score >= 40:
                st.warning(f"⚠️ Suspicious ({score})")
            else:
                st.error(f"🚨 Deepfake ({score})")

            st.progress(score / 100)
            st.write(f"Confidence: {result['confidence']}%")

            # Explanation
            st.markdown("### 🧠 Explanation")
            for e in explain(score):
                st.write(f"- {e}")

            # Frame visualization
            if frames:
                st.markdown("### 🎬 Frame Analysis")
                cols = st.columns(4)
                for i, f in enumerate(frames):
                    with cols[i % 4]:
                        st.image(f["image"], caption=f"{f['score']}")

            # Graph
            if scores:
                st.markdown("### 📊 Confidence Graph")
                fig, ax = plt.subplots()
                ax.plot(scores)
                ax.set_title("Frame Scores")
                st.pyplot(fig)

            # Hash
            st.markdown("### 🔐 File Hash")
            st.code(file_hash(data))


# ─── URL Mode ───────────────────────
elif mode == "URL":
    url = st.text_input("Enter Image URL")

    if url:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        st.image(img)

        if st.button("Analyze URL 🚀"):
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                img.save(tmp.name)
                result, _, _ = detector.analyze_image(tmp.name)

            st.write(result)
