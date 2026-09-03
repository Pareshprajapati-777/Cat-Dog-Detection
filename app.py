import streamlit as st
import torch
from PIL import Image
import os
import random
from model_utils import load_classifier, predict_image

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Cat vs Dog AI Detector",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Styling (Dark Glassmorphism UI)
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 50% 0%, #1e1e38 0%, #0b0b14 100%);
        color: #f1f5f9;
    }

    .main-header {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
    }

    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a855f7 0%, #6366f1 50%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }

    .sub-title {
        color: #94a3b8;
        font-size: 1.1rem;
        font-weight: 400;
    }

    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
    }

    .result-badge {
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }

    .cat-badge {
        color: #ec4899;
        text-shadow: 0 0 20px rgba(236, 72, 153, 0.4);
    }

    .dog-badge {
        color: #3b82f6;
        text-shadow: 0 0 20px rgba(59, 130, 246, 0.4);
    }

    .unknown-badge {
        color: #f59e0b;
        text-shadow: 0 0 20px rgba(245, 158, 11, 0.4);
    }

    .confidence-text {
        font-size: 1.3rem;
        font-weight: 600;
        color: #cbd5e1;
        margin-bottom: 1.2rem;
    }

    .object-tag {
        display: inline-block;
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.35);
        color: #fbbf24;
        padding: 0.4rem 1rem;
        border-radius: 9999px;
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 1.2rem;
    }

    .metric-row {
        display: flex;
        justify-content: space-around;
        margin-top: 1rem;
        background: rgba(15, 23, 42, 0.6);
        padding: 1rem;
        border-radius: 12px;
    }

    .metric-box {
        text-align: center;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
    }

    div[data-testid="stFileUploader"] {
        background: rgba(30, 41, 59, 0.5);
        border: 2px dashed #6366f1;
        border-radius: 16px;
        padding: 1.5rem;
    }

    div[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Model Resource (Cached)
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading PyTorch AI Engine...")
def get_model():
    return load_classifier("cat_dog_model.pth")

model, model_type = get_model()

# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/cat-footprint.png", width=70)
    st.title("🐾 Control Panel")
    st.markdown("---")
    
    st.markdown("### 🤖 Model Engine")
    if model_type == "fine_tuned":
        st.success("✅ Fine-Tuned + Open-World Detector")
    else:
        st.info("⚡ MobileNetV2 Open-World Detector")
        
    st.markdown("---")
    st.markdown("### 🖼️ Quick Sample Test")
    st.caption("Select a sample image to test instantly:")
    
    sample_cat_dir = "archive/Cats"
    sample_dog_dir = "archive/Dogs"
    sample_unknown_dir = "archive/Unknown"
    
    sample_cat = None
    sample_dog = None
    sample_unknown = None
    
    if os.path.exists(sample_cat_dir) and os.listdir(sample_cat_dir):
        cat_files = os.listdir(sample_cat_dir)
        sample_cat = os.path.join(sample_cat_dir, cat_files[0])
        
    if os.path.exists(sample_dog_dir) and os.listdir(sample_dog_dir):
        dog_files = os.listdir(sample_dog_dir)
        sample_dog = os.path.join(sample_dog_dir, dog_files[0])

    if os.path.exists(sample_unknown_dir) and os.listdir(sample_unknown_dir):
        unknown_files = os.listdir(sample_unknown_dir)
        sample_unknown = os.path.join(sample_unknown_dir, unknown_files[0])
        
    if "current_sample" not in st.session_state:
        st.session_state.current_sample = None

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        if sample_cat and st.button("🐱 Cat"):
            st.session_state.current_sample = sample_cat
    with col_s2:
        if sample_dog and st.button("🐶 Dog"):
            st.session_state.current_sample = sample_dog
    with col_s3:
        if sample_unknown and st.button("🚗 Other"):
            st.session_state.current_sample = sample_unknown

    if st.session_state.current_sample is not None:
        if st.button("🔄 Clear Sample Selection"):
            st.session_state.current_sample = None
            st.rerun()

    st.markdown("---")
    st.markdown("### 🛡️ Smart Classification")
    st.caption("Detects Cat 🐱, Dog 🐶, or automatically flags any other object as **Unknown** ❓.")
    st.caption("Powered by PyTorch 2.13 & Streamlit")

# ---------------------------------------------------------
# Main UI Header
# ---------------------------------------------------------
st.markdown("""
<div class="main-header">
    <div class="main-title">🐾 Cat vs Dog AI Detector</div>
    <div class="sub-title">Upload an image or pick a sample to detect Cat 🐱, Dog 🐶, or Unknown ❓ with instant AI analysis</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Input Section
# ---------------------------------------------------------
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### 📥 Input Image")
    uploaded_file = st.file_uploader(
        "Choose a JPG, JPEG, or PNG image",
        type=["jpg", "jpeg", "png"]
    )
    
    img_to_process = None
    
    if uploaded_file is not None:
        img_to_process = Image.open(uploaded_file)
        caption_text = f"Uploaded: {uploaded_file.name}"
    elif st.session_state.current_sample is not None:
        img_to_process = Image.open(st.session_state.current_sample)
        caption_text = f"Sample: {os.path.basename(st.session_state.current_sample)}"
        
    if img_to_process is not None:
        st.image(img_to_process, caption=caption_text, use_container_width=True)
    else:
        st.info("👆 Upload an image using the box above, or choose a sample from the sidebar!")

# ---------------------------------------------------------
# Prediction & Analysis Section
# ---------------------------------------------------------
with col_right:
    st.markdown("### 📊 Prediction & Analysis")
    
    if img_to_process is not None:
        with st.spinner("Analyzing image features..."):
            res = predict_image(img_to_process, model, model_type)
            
        if res["label"] == "Dog":
            badge_class = "dog-badge"
        elif res["label"] == "Cat":
            badge_class = "cat-badge"
        else:
            badge_class = "unknown-badge"
        
        if res["label"] == "Unknown":
            st.markdown(f"""
            <div class="glass-card">
                <div class="sub-title">Classification Result</div>
                <div class="result-badge {badge_class}">{res["emoji"]} {res["label"]}</div>
                <div class="confidence-text">Neither Cat nor Dog Detected</div>
                <div class="object-tag">🔍 Detected: {res.get('detected_object', 'Non-Cat/Dog')}</div>
                <div class="metric-row">
                    <div class="metric-box">
                        <div class="metric-label">🐱 Cat Match</div>
                        <div class="metric-value" style="color: #ec4899;">{res["cat_score"]:.1f}%</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">🐶 Dog Match</div>
                        <div class="metric-value" style="color: #3b82f6;">{res["dog_score"]:.1f}%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.warning(f"⚠️ **Notice:** This image does not contain a Cat or Dog. The AI detected features resembling **{res.get('detected_object', 'an unknown object')}**.")
        else:
            st.markdown(f"""
            <div class="glass-card">
                <div class="sub-title">Detected Class</div>
                <div class="result-badge {badge_class}">{res["emoji"]} {res["label"]}</div>
                <div class="confidence-text">Confidence: <strong>{res["confidence"]:.1f}%</strong></div>
                <div class="metric-row">
                    <div class="metric-box">
                        <div class="metric-label">🐱 Cat Probability</div>
                        <div class="metric-value" style="color: #ec4899;">{res["cat_score"]:.1f}%</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">🐶 Dog Probability</div>
                        <div class="metric-value" style="color: #3b82f6;">{res["dog_score"]:.1f}%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Cat vs Dog Probability Distribution**")
            st.progress(float(res["dog_score"] / 100.0), text=f"🐶 Dog ({res['dog_score']:.1f}%) | 🐱 Cat ({res['cat_score']:.1f}%)")
        
    else:
        st.markdown("""
        <div class="glass-card" style="padding: 3rem 1.5rem;">
            <div style="font-size: 3rem;">🔍</div>
            <h4 style="color: #94a3b8; margin-top: 1rem;">Waiting for Image Input</h4>
            <p style="color: #64748b; font-size: 0.9rem;">Upload an image or pick a sample to view real-time AI classification scores.</p>
        </div>
        """, unsafe_allow_html=True)
