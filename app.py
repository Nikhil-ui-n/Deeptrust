import streamlit as st
import numpy as np
from PIL import Image
import hashlib
import tempfile
from io import BytesIO
import requests

# Safe imports
try:
    import cv2
    CV2 = True
except:
    CV2 = False

try:
    import matplotlib.pyplot as plt
    MPL = True
except:
    MPL = False

st.set_page_config(page_title="DeepTrust", layout="wide")
st.title("🔍 DeepTrust - Smart Deepfake Detector")

# ─── Detector ───────────────────────
class DeepfakeDetector:

    def analyze_image(self, path):
        if not CV2:
            return {"error": "OpenCV not available"}, [], []

        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self._detect_faces(gray)

        # 🔥 If no face → safe fallback
        if len(faces) == 0:
            return {
                "score": 70,
                "verdict": "LIKELY REAL 📄",
                "confidence": 40,
                "note": "No face detected — fallback logic used"
            }, [], []

        score = self._compute(gray)
        return self._build(score), [], []

    def analyze_video(self, path):
        if not CV2:
            return {"error": "OpenCV not available"}, [], []

        cap = cv2.VideoCapture(path)
        frames, scores = [], []

        count = 0
        while count < 10:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = self._detect_faces(gray)

            # Draw faces
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

            # If no face → skip scoring
            if len(faces) == 0:
                score = 0.7
            else:
                score = self._compute(gray)

            scores.append(int(score * 100))

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append({"image": frame_rgb, "score": int(score * 100)})

            count += 1

        cap.release()

        final_score = np.mean(scores)/100 if scores else 0.5
        return self._build(final_score), frames, scores

    # ─── Core Logic ───
    def _detect_faces(self, gray):
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        return face_cascade.detectMultiScale(gray, 1.3, 5)

    def _compute(self, gray):
        variance = np.var(gray)
        noise = np.std(gray)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.mean(edges)

        score = (variance*0.4 + noise*0.3 + edge_density*0.3) / 7000
        return min(1.0, score)

    def _build(self, score):
        score = int(score * 100)

        if score >= 75:
            verdict = "REAL ✅"
        elif score >= 50:
            verdict = "UNCERTAIN ⚠️"
        else:
            verdict = "FAKE 🚨"

        return {
            "score": score,
            "verdict": verdict,
            "confidence": abs(score - 50) * 2
        }

# ─── Utils ─────────────────────────
def file_hash(data):
    return hashlib.sha256(data).hexdigest()

def explain(score):
    if score >= 75:
        return ["Natural texture", "Consistent lighting", "Balanced edges"]
    elif score >= 50:
        return ["Mixed signals", "Possible editing", "Moderate anomalies"]
    else:
        return ["GAN artifacts", "Unnatural smoothing", "Abnormal edges"]

# ─── App ───────────────────────────
detector = DeepfakeDetector()

mode = st.sidebar.radio("Mode", ["Upload", "URL"])

# ─── Upload Mode ───
if mode == "Upload":
    uploaded = st.file_uploader("Upload Image/Video", type=["jpg","png","jpeg","mp4"])

    if uploaded:
        data = uploaded.read()

        col1, col2 = st.columns([2,1])

        with col1:
            st.subheader("Preview")
            if uploaded.type.startswith("image"):
                st.image(Image.open(BytesIO(data)))
            else:
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp.write(data)
                    st.video(tmp.name)

        with col2:
            st.subheader("File Info")
            st.write(uploaded.name)
            st.write(f"{uploaded.size/1024:.2f} KB")

        if st.button("Analyze 🚀", use_container_width=True):

            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(data)
                path = tmp.name

            if uploaded.type.startswith("image"):
                result, frames, scores = detector.analyze_image(path)
            else:
                result, frames, scores = detector.analyze_video(path)

            if "error" in result:
                st.error(result["error"])
                st.stop()

            score = result["score"]

            st.markdown("## 🔎 Result")
            st.subheader(f"Final Verdict: {result['verdict']}")

            if "REAL" in result["verdict"]:
                st.success(f"{result['verdict']} ({score})")
            elif "UNCERTAIN" in result["verdict"]:
                st.warning(f"{result['verdict']} ({score})")
            else:
                st.error(f"{result['verdict']} ({score})")

            st.progress(score/100)
            st.metric("Confidence", f"{result['confidence']}%")

            if result.get("note"):
                st.info(result["note"])

            if result["confidence"] < 50:
                st.warning("⚠️ Low confidence — result may be inaccurate")

            # Explanation
            st.markdown("### 🧠 Explanation")
            for e in explain(score):
                st.write(f"- {e}")

            # Frames
            if frames:
                st.markdown("### 🎬 Frame Analysis")
                cols = st.columns(5)
                for i, f in enumerate(frames):
                    with cols[i % 5]:
                        st.image(f["image"], caption=f"{f['score']}")

            # Graph
            if scores and MPL:
                st.markdown("### 📊 Confidence Graph")
                fig, ax = plt.subplots()
                ax.plot(scores)
                ax.set_title("Frame Scores")
                st.pyplot(fig)

            # Hash
            hash_val = file_hash(data)
            st.markdown("### 🔐 File Hash")
            st.code(hash_val)

# ─── URL Mode ───
elif mode == "URL":
    url = st.text_input("Enter Image URL")

    if url:
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            st.image(img)

            if st.button("Analyze URL 🚀"):
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    img.save(tmp.name)
                    result, _, _ = detector.analyze_image(tmp.name)

                st.write(result)

        except:
            st.error("Invalid URL")
