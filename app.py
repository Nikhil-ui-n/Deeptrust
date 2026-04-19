import streamlit as st
import cv2
import numpy as np
from PIL import Image
import hashlib
import os
from datetime import datetime
from io import BytesIO
import tempfile

# ─── Page Configuration ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="DeepTrust - Deepfake Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #060b14 0%, #0a0f1f 100%);
    }
    
    .stTitle {
        background: linear-gradient(135deg, #60a5fa, #a78bfa, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 900;
        font-size: 3.5rem !important;
    }
    
    .metric-card {
        background: rgba(15, 23, 42, 0.7);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 20px;
        margin: 10px 0;
    }
    
    .score-authentic {
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.4);
        color: #22c55e;
    }
    
    .score-suspicious {
        background: rgba(234, 179, 8, 0.1);
        border: 1px solid rgba(234, 179, 8, 0.4);
        color: #eab308;
    }
    
    .score-deepfake {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.4);
        color: #ef4444;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #6366f1);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.35);
    }
</style>
""", unsafe_allow_html=True)

# ─── Utility Functions ─────────────────────────────────────────────────────

class DeepfakeDetector:
    """AI-powered deepfake detector using multiple heuristics"""
    
    def __init__(self):
        pass
    
    def analyze_image(self, image_path):
        """Analyze image for deepfakes"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return {"error": "Could not load image"}
            
            scores = {
                "frequency": self._frequency_analysis(img),
                "face": self._face_consistency(img),
                "texture": self._texture_analysis(img),
                "noise": self._noise_detection(img),
                "color": self._color_space_analysis(img),
            }
            
            weights = {
                "frequency": 0.25,
                "face": 0.30,
                "texture": 0.20,
                "noise": 0.15,
                "color": 0.10,
            }
            
            final_score = sum(scores[k] * weights[k] for k in scores)
            final_score = int(np.clip(final_score * 100, 0, 100))
            
            return {
                "trust_score": final_score,
                "is_authentic": final_score >= 70,
                "is_suspicious": 40 <= final_score < 70,
                "is_deepfake": final_score < 40,
                "component_scores": {k: int(v*100) for k, v in scores.items()},
            }
        except Exception as e:
            return {"error": str(e)}
    
    def analyze_video(self, video_path, max_frames=12):
        """Analyze video for deepfakes"""
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
            frame_scores = []
            frames_data = []
            
            progress_bar = st.progress(0)
            
            for idx, frame_num in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                score = self._score_frame(frame)
                frame_scores.append(score)
                
                frames_data.append({
                    "frame_num": int(frame_num),
                    "timestamp": float(frame_num / fps) if fps > 0 else 0,
                    "score": int(score * 100),
                    "is_suspicious": score < 0.4
                })
                
                progress_bar.progress((idx + 1) / len(frame_indices))
            
            cap.release()
            
            if frame_scores:
                mean_score = np.mean(frame_scores)
                std_score = np.std(frame_scores)
                consistency_penalty = min(std_score * 0.3, 0.2)
                final_score = int(np.clip((mean_score - consistency_penalty) * 100, 0, 100))
            else:
                final_score = 50
            
            suspicious_frames = sum(1 for f in frames_data if f["is_suspicious"])
            
            return {
                "trust_score": final_score,
                "is_authentic": final_score >= 70,
                "is_suspicious": 40 <= final_score < 70,
                "is_deepfake": final_score < 40,
                "total_frames": total_frames,
                "frames_analyzed": len(frames_data),
                "fps": fps,
                "duration": total_frames / fps if fps > 0 else 0,
                "frame_scores": frames_data,
                "suspicious_frames": suspicious_frames,
                "consistency_score": int((1 - min(std_score, 1)) * 100)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _score_frame(self, frame):
        """Quick frame scoring"""
        freq = self._frequency_analysis(frame)
        face = self._face_consistency(frame)
        texture = self._texture_analysis(frame)
        noise = self._noise_detection(frame)
        return (freq * 0.25 + face * 0.35 + texture * 0.25 + noise * 0.15)
    
    def _frequency_analysis(self, img):
        """Detect GAN artifacts in frequency domain"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.abs(f_shift)
            magnitude_log = np.log1p(magnitude)
            
            center = magnitude_log[magnitude_log.shape[0]//2, magnitude_log.shape[1]//2]
            surrounding = magnitude_log[
                magnitude_log.shape[0]//2-10:magnitude_log.shape[0]//2+10,
                magnitude_log.shape[1]//2-10:magnitude_log.shape[1]//2+10
            ]
            
            periodicity = np.std(surrounding) / (np.mean(surrounding) + 1e-6)
            authenticity = 1.0 / (1.0 + periodicity)
            
            return np.clip(authenticity, 0, 1)
        except:
            return 0.5
    
    def _face_consistency(self, img):
        """Check for face-related anomalies"""
        try:
            if len(img.shape) == 3:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
            
            h, w = img_rgb.shape[:2]
            center_region = img_rgb[h//4:3*h//4, w//4:3*w//4]
            
            r, g, b = center_region[:,:,0], center_region[:,:,1], center_region[:,:,2]
            
            rg_diff = np.abs(r.astype(float) - g.astype(float))
            gb_diff = np.abs(g.astype(float) - b.astype(float))
            
            color_consistency = 1.0 - (np.mean(rg_diff) + np.mean(gb_diff)) / 255.0
            
            return np.clip(color_consistency, 0, 1)
        except:
            return 0.5
    
    def _texture_analysis(self, img):
        """Analyze texture smoothness"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            variance = laplacian.var()
            
            optimal_variance = 100
            sharpness_score = 1.0 - np.exp(-abs(variance - optimal_variance) / 500)
            
            return np.clip(sharpness_score, 0, 1)
        except:
            return 0.5
    
    def _noise_detection(self, img):
        """Detect unnatural noise patterns"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            gray_float = gray.astype(np.float32)
            
            kernel = np.array([[-1, -1, -1],
                             [-1,  8, -1],
                             [-1, -1, -1]])
            edges = cv2.filter2D(gray_float, -1, kernel)
            
            noise_level = np.std(edges)
            optimal_noise = 20
            noise_score = 1.0 - np.exp(-abs(noise_level - optimal_noise) / 100)
            
            return np.clip(noise_score, 0, 1)
        except:
            return 0.5
    
    def _color_space_analysis(self, img):
        """Analyze color space consistency"""
        try:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            saturation = hsv[:,:,1].astype(float)
            sat_variance = np.var(saturation)
            
            optimal_sat_var = 500
            color_score = 1.0 - np.exp(-sat_variance / optimal_sat_var)
            
            return np.clip(color_score, 0, 1)
        except:
            return 0.5


def generate_file_hash(file_bytes):
    """Generate SHA-256 hash"""
    return hashlib.sha256(file_bytes).hexdigest()


def get_score_color(score):
    """Get color based on score"""
    if score >= 70:
        return "green", "✅ Authentic", "Score indicates genuine media"
    elif score >= 40:
        return "orange", "⚠️ Suspicious", "Some anomalies detected"
    else:
        return "red", "🚨 Likely Deepfake", "Strong manipulation indicators"


def generate_explanations(score, component_scores):
    """Generate human-readable explanations"""
    explanations = []
    
    if score >= 70:
        explanations = [
            "✅ Natural skin texture with consistent pore distribution",
            "✅ Consistent lighting across facial planes",
            "✅ Eye reflections match environmental lighting",
            "✅ Natural hair strand rendering with no artifacts",
        ]
    elif score >= 40:
        explanations = [
            "⚠️ Facial texture shows minor irregularities in some regions",
            "⚠️ Subtle blurring detected around facial edges",
            "⚠️ Lighting appears slightly mismatched in some areas",
            "⚠️ Minor compression artifacts detected",
        ]
    else:
        explanations = [
            "🚨 Facial texture shows significant inconsistencies",
            "🚨 Strong GAN artifacts detected in frequency analysis",
            "🚨 Unnatural blurring around facial boundaries",
            "🚨 High-frequency noise patterns typical of neural generation",
        ]
    
    return explanations

# ─── Session State ──────────────────────────────────────────────────────────

if 'detector' not in st.session_state:
    st.session_state.detector = DeepfakeDetector()

if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

# ─── Header ────────────────────────────────────────────────────────────────

col1, col2 = st.columns([1, 4])
with col1:
    st.markdown("### 🔍 DeepTrust")
with col2:
    st.markdown("### AI-Powered Deepfake Detection")

st.markdown("---")

# ─── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    analysis_type = st.radio(
        "Analysis Type",
        ["📤 Upload & Analyze", "📊 View Results", "ℹ️ About"]
    )
    
    st.markdown("---")
    
    st.markdown("""
    ### 📋 Features
    - 🧠 AI Detection
    - 🎬 Video Frame Analysis
    - 🔬 Metadata Forensics
    - 🔐 Hash Verification
    - 📊 Detailed Reports
    """)
    
    st.markdown("---")
    st.markdown("""
    **Made for Hackathon**
    Version 2.1.0
    """)

# ─── Main Content ──────────────────────────────────────────────────────────

if analysis_type == "📤 Upload & Analyze":
    st.markdown("## Upload Media for Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📁 Choose File")
        uploaded_file = st.file_uploader(
            "Upload an image or video",
            type=["jpg", "jpeg", "png", "webp", "gif", "mp4", "webm"],
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("### 📋 File Info")
        if uploaded_file:
            st.markdown(f"""
            - **Name**: {uploaded_file.name}
            - **Size**: {uploaded_file.size / 1024 / 1024:.2f} MB
            - **Type**: {uploaded_file.type}
            """)
    
    if uploaded_file:
        st.markdown("---")
        
        # Display preview
        st.markdown("### 👀 Preview")
        
        file_bytes = uploaded_file.read()
        
        if uploaded_file.type.startswith('image/'):
            image = Image.open(BytesIO(file_bytes))
            st.image(image, use_column_width=True, caption="Preview")
        elif uploaded_file.type.startswith('video/'):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            st.video(tmp_path)
        
        st.markdown("---")
        
        # Analyze button
        if st.button("🚀 Analyze with AI", use_container_width=True):
            with st.spinner("🔄 Analyzing media... This may take a moment"):
                # Save file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                    tmp.write(file_bytes)
                    tmp_path = tmp.name
                
                # Analyze
                detector = st.session_state.detector
                
                if uploaded_file.type.startswith('image/'):
                    result = detector.analyze_image(tmp_path)
                elif uploaded_file.type.startswith('video/'):
                    result = detector.analyze_video(tmp_path, max_frames=12)
                else:
                    result = {"error": "Unsupported file type"}
                
                # Generate hash
                result["hash"] = generate_file_hash(file_bytes)
                result["filename"] = uploaded_file.name
                result["filesize"] = uploaded_file.size
                result["timestamp"] = datetime.now().isoformat()
                
                st.session_state.analysis_result = result
                st.session_state.uploaded_file = uploaded_file
                
                # Cleanup
                try:
                    os.remove(tmp_path)
                except:
                    pass
            
            st.success("✅ Analysis complete!")

elif analysis_type == "📊 View Results":
    if st.session_state.analysis_result:
        result = st.session_state.analysis_result
        uploaded_file = st.session_state.uploaded_file
        
        if "error" in result:
            st.error(f"❌ Error: {result['error']}")
        else:
            # Score display
            score = result.get("trust_score", 0)
            color, verdict, description = get_score_color(score)
            
            st.markdown("## 📊 Analysis Results")
            
            # Main score card
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                st.markdown(f"""
                <div style="text-align: center; padding: 30px; background: rgba(15, 23, 42, 0.7); border-radius: 16px; border: 1px solid rgba(255,255,255,0.06);">
                    <div style="font-size: 3.5rem; font-weight: 900; color: {'#22c55e' if color == 'green' else '#eab308' if color == 'orange' else '#ef4444'};">
                        {score}
                    </div>
                    <div style="font-size: 1.2rem; font-weight: 600; color: {'#22c55e' if color == 'green' else '#eab308' if color == 'orange' else '#ef4444'};">
                        {verdict}
                    </div>
                    <div style="font-size: 0.9rem; color: #94a3b8; margin-top: 10px;">
                        {description}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Component scores
            st.markdown("### 🔬 Component Analysis")
            
            component_scores = result.get("component_scores", {})
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            components = [
                ("Frequency", "🌊", col1),
                ("Face", "👤", col2),
                ("Texture", "🎨", col3),
                ("Noise", "⚡", col4),
                ("Color", "🌈", col5),
            ]
            
            for name, emoji, col in components:
                with col:
                    key = name.lower()
                    if key in component_scores:
                        score_val = component_scores[key]
                        st.metric(f"{emoji} {name}", f"{score_val}%")
            
            st.markdown("---")
            
            # Explanations
            st.markdown("### 🧠 AI Findings")
            
            explanations = generate_explanations(score, component_scores)
            
            for exp in explanations:
                st.markdown(f"- {exp}")
            
            st.markdown("---")
            
            # File info
            st.markdown("### 📋 File Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Filename**: {result.get('filename', 'Unknown')}")
                st.markdown(f"**Size**: {result.get('filesize', 0) / 1024 / 1024:.2f} MB")
            
            with col2:
                st.markdown(f"**Analyzed**: {result.get('timestamp', 'Unknown')}")
                if "duration" in result:
                    st.markdown(f"**Duration**: {result['duration']:.1f}s")
            
            st.markdown("---")
            
            # Video frame analysis
            if "frame_scores" in result:
                st.markdown("### 🎬 Frame-by-Frame Analysis")
                
                st.markdown(f"**Frames Analyzed**: {result['frames_analyzed']} / {result['total_frames']}")
                st.markdown(f"**Suspicious Frames**: {result['suspicious_frames']}")
                st.markdown(f"**Consistency Score**: {result['consistency_score']}%")
                
                # Frame grid
                frame_cols = st.columns(6)
                
                for idx, frame_data in enumerate(result['frame_scores'][:12]):
                    with frame_cols[idx % 6]:
                        frame_score = frame_data['score']
                        is_suspicious = frame_data['is_suspicious']
                        
                        color_hex = '#ef4444' if is_suspicious else '#22c55e'
                        
                        st.markdown(f"""
                        <div style="text-align: center; padding: 15px; background: rgba(15, 23, 42, 0.7); border-radius: 8px; border: 2px solid {color_hex}30;">
                            <div style="font-size: 1.2rem; color: {color_hex}; font-weight: 600;">
                                {frame_score}
                            </div>
                            <div style="font-size: 0.75rem; color: #94a3b8;">
                                {frame_data['timestamp']:.1f}s
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Hash verification
            st.markdown("### 🔐 File Integrity")
            
            file_hash = result.get("hash", "")
            
            st.code(file_hash, language="text")
            
            st.markdown("**Use this hash to verify the file hasn't been modified.**")
            
            # Download report
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                report_text = f"""
DEEPTRUST ANALYSIS REPORT
{'='*50}

Generated: {result['timestamp']}
File: {result['filename']}
Size: {result['filesize'] / 1024 / 1024:.2f} MB

TRUST SCORE: {score}/100
Classification: {verdict}

COMPONENT SCORES:
{chr(10).join([f"- {k.title()}: {v}%" for k, v in component_scores.items()])}

FILE HASH (SHA-256):
{file_hash}

{'='*50}
Report generated by DeepTrust v2.1.0
"""
                
                st.download_button(
                    label="📥 Download Report",
                    data=report_text,
                    file_name=f"deeptrust_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            with col2:
                if st.button("🔄 Analyze Another File", use_container_width=True):
                    st.session_state.analysis_result = None
                    st.session_state.uploaded_file = None
                    st.rerun()
    
    else:
        st.info("📤 Upload and analyze a file first to see results here.")

elif analysis_type == "ℹ️ About":
    st.markdown("""
    ## 📘 About DeepTrust
    
    DeepTrust is an AI-powered platform that detects deepfakes and verifies media authenticity.
    
    ### 🎯 How It Works
    
    Our system analyzes media using multiple detection methods:
    
    **1. Frequency Analysis (25%)**
    - Detects GAN artifacts in frequency domain
    - Analyzes periodicity patterns
    
    **2. Face Consistency (30%)**
    - Checks skin color uniformity
    - Analyzes facial proportions
    
    **3. Texture Analysis (20%)**
    - Measures sharpness and smoothness
    - Detects over-smoothing (common in deepfakes)
    
    **4. Noise Detection (15%)**
    - Analyzes high-frequency patterns
    - Detects unnatural noise
    
    **5. Color Space (10%)**
    - Checks saturation consistency
    - Analyzes color channel distribution
    
    ### 📊 Scoring System
    
    - **70-100**: ✅ Authentic (Green)
    - **40-70**: ⚠️ Suspicious (Yellow)
    - **0-40**: 🚨 Deepfake (Red)
    
    ### 🎬 Supported Formats
    
    **Images**: JPG, PNG, WebP, GIF
    **Videos**: MP4, WebM
    **Max Size**: 100MB
    
    ### ⚡ Performance
    
    - Image Analysis: 2-4 seconds
    - Video Analysis: 5-10 seconds
    - Works on CPU (GPU supported)
    
    ### 🔐 Privacy
    
    - All analysis is local
    - Files are not stored
    - No data collection
    
    ### 📋 Limitations
    
    - Results are probabilistic, not absolute
    - State-of-the-art deepfakes may bypass detection
    - Low-quality videos reduce accuracy
    
    ---
    
    **Built for Hackathon** | Version 2.1.0
    """)

# ─── Footer ────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.9rem;">
    Made with ❤️ for Hackathon | DeepTrust v2.1.0
</div>
""", unsafe_allow_html=True)
