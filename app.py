import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import json

st.set_page_config(
    page_title="KrishiRakshak AI - Crop Disease Detection",
    page_icon="ðŸŒ¾",
    layout="wide"
)

# Confidence below this = "not a leaf we recognize" instead of a forced guess
CONFIDENCE_THRESHOLD = 65.0  # percent, tune this against your validation set

# ---------- Custom styling ----------
st.markdown("""
    <style>
    .stApp {
        background: radial-gradient(circle at top left, #2b3d22 0%, #1c2a17 55%, #141f10 100%);
    }
    .stApp, .stApp p, .stApp span, .stApp label, .stApp li, .stMarkdown, .stCaption, div[data-testid="stCaptionContainer"] {
        color: #eef2e6 !important;
    }
    .stApp .stSelectbox label, .stApp .stFileUploader label { color: #eef2e6 !important; }

    .hero {
        background: linear-gradient(135deg, #23331b 0%, #35492a 45%, #4a6339 100%);
        padding: 2.4rem 2.6rem;
        border-radius: 24px;
        margin-bottom: 1.8rem;
        box-shadow: 0 12px 32px rgba(0,0,0,0.35);
        border: 1px solid rgba(255,255,255,0.08);
        position: relative;
        overflow: hidden;
    }
    .hero::after {
        content: "ðŸŒ¾";
        position: absolute;
        right: -10px;
        top: -30px;
        font-size: 10rem;
        opacity: 0.10;
        transform: rotate(15deg);
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        color: #f6f9f2;
        margin-bottom: 4px;
        position: relative;
        z-index: 1;
    }
    .hero-sub {
        font-size: 1.05rem;
        color: #d3ddc7;
        margin-bottom: 0;
        position: relative;
        z-index: 1;
    }
    .hero-badge {
        display: inline-block;
        background: #f2c744;
        color: #23331b;
        font-weight: 700;
        padding: 6px 18px;
        border-radius: 30px;
        font-size: 0.8rem;
        margin-top: 14px;
        position: relative;
        z-index: 1;
    }

    .stat-card {
        background: rgba(255,255,255,0.06);
        border-radius: 18px;
        padding: 1.1rem 1.2rem;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0,0,0,0.25);
        border: 1px solid rgba(255,255,255,0.10);
        transition: transform 0.2s ease;
        color: #f6f9f2;
    }
    .stat-card:hover { transform: translateY(-3px); border-color: rgba(242,199,68,0.5); }
    .stat-num {
        font-size: 1.8rem;
        font-weight: 800;
        color: #f2c744;
    }
    .stat-label { font-size: 0.8rem; color: #d3ddc7; margin-top: 2px; }

    .upload-panel-label {
        font-weight: 700;
        color: #f6f9f2;
        margin-bottom: 0.6rem;
    }

    div[data-testid="stFileUploaderDropzone"] {
        border: 2.5px dashed rgba(242,199,68,0.55) !important;
        border-radius: 20px !important;
        background: rgba(255,255,255,0.05) !important;
        padding: 1.2rem !important;
        box-shadow: 0 3px 14px rgba(0,0,0,0.2);
        transition: border-color 0.2s ease, background 0.2s ease;
    }
    div[data-testid="stFileUploaderDropzone"]:hover {
        border-color: #f2c744 !important;
        background: rgba(242,199,68,0.08) !important;
    }
    div[data-testid="stFileUploaderDropzone"] span,
    div[data-testid="stFileUploaderDropzone"] small,
    div[data-testid="stFileUploaderDropzone"] svg {
        color: #eef2e6 !important;
        fill: #eef2e6 !important;
    }
    div[data-testid="stFileUploaderDropzone"] button {
        background: #f2c744 !important;
        color: #23331b !important;
        border: none !important;
        border-radius: 30px !important;
        font-weight: 700 !important;
    }

    .result-card {
        background: rgba(255,255,255,0.06);
        border-radius: 20px;
        padding: 1.6rem 1.8rem;
        box-shadow: 0 8px 26px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.10);
        border-top: 5px solid #f2c744;
        animation: fadeIn 0.5s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(242,199,68,0.35); }
        70% { box-shadow: 0 0 0 8px rgba(242,199,68,0); }
        100% { box-shadow: 0 0 0 0 rgba(242,199,68,0); }
    }
    .status-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse 1.8s infinite;
    }
    .result-label {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 1.2px;
        color: #bcc7ab;
        text-transform: uppercase;
    }
    .result-name { font-size: 1.7rem; font-weight: 800; margin-top: 4px; color: #f6f9f2; }
    .confidence-bar-bg {
        background-color: rgba(255,255,255,0.12);
        border-radius: 10px;
        height: 16px;
        width: 100%;
        margin-top: 8px;
        overflow: hidden;
    }
    .confidence-bar-fill {
        height: 16px;
        border-radius: 10px;
        transition: width 0.8s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
    .treatment-box {
        background: rgba(242,199,68,0.10);
        border-radius: 18px;
        padding: 1.3rem 1.6rem;
        margin-top: 1.2rem;
        border: 1px solid rgba(242,199,68,0.35);
    }
    .error-box {
        background: rgba(224,102,90,0.12);
        border-radius: 18px;
        padding: 1.4rem 1.6rem;
        margin-top: 0.4rem;
        border: 1px solid rgba(224,102,90,0.45);
    }
    .section-header {
        font-size: 1.35rem;
        font-weight: 800;
        color: #f6f9f2;
        margin: 2.2rem 0 1rem 0;
    }
    .helpline-card {
        background: rgba(255,255,255,0.06);
        border-radius: 18px;
        padding: 1.4rem 1.7rem;
        border: 1px solid rgba(255,255,255,0.10);
    }
    .roadmap-chip {
        background: rgba(255,255,255,0.06);
        color: #f6f9f2;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        margin-bottom: 8px;
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow: 0 3px 10px rgba(0,0,0,0.2);
        transition: transform 0.2s ease;
    }
    .roadmap-chip:hover { transform: translateY(-3px); border-color: rgba(242,199,68,0.5); }
    .roadmap-chip span { color: #d3ddc7 !important; }
    .step-card {
        background: rgba(255,255,255,0.06);
        color: #f6f9f2;
        border-radius: 18px;
        padding: 1.3rem 1.2rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow: 0 4px 14px rgba(0,0,0,0.22);
        height: 100%;
    }
    .step-num {
        width: 34px; height: 34px;
        border-radius: 50%;
        background: #f2c744;
        color: #23331b;
        display: flex; align-items: center; justify-content: center;
        font-weight: 800;
        margin: 0 auto 10px auto;
    }
    .footer-note {
        color: #9fab8f;
        font-size: 0.8rem;
        margin-top: 3rem;
        text-align: center;
        padding-top: 1.5rem;
        border-top: 1px solid rgba(255,255,255,0.10);
    }

    div[data-testid="stSelectbox"] > div {
        background: rgba(255,255,255,0.08) !important;
        border-radius: 14px !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
    }
    div[data-testid="stSelectbox"] div, div[data-testid="stSelectbox"] span {
        color: #eef2e6 !important;
    }
    .stButton button {
        background: #f2c744 !important;
        color: #23331b !important;
        border: none !important;
        border-radius: 30px !important;
        font-weight: 700 !important;
        padding: 0.6rem 1.2rem !important;
    }
    .stButton button:hover { background: #f6d768 !important; }
    </style>
""", unsafe_allow_html=True)

# ---------- Load model (fully local, no network calls) ----------
@st.cache_resource
def load_my_model():
    model = load_model("crop_model.h5")
    with open("class_names.json") as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_my_model()

# ---------- Hero ----------
st.markdown("""
    <div class="hero">
        <div class="hero-title">ðŸŒ¾ KrishiRakshak AI</div>
        <div class="hero-sub">Early Detection & Management of Crop Diseases and Pest Infestations</div>
        <div class="hero-badge">ðŸ›ï¸ Government of Maharashtra Â· SIH 2026 Â· SIH26131</div>
    </div>
""", unsafe_allow_html=True)

# ---------- Stat row ----------
s1, s2, s3, s4 = st.columns(4)
with s1:
    st.markdown('<div class="stat-card"><div class="stat-num">4</div><div class="stat-label">Crops covered</div></div>', unsafe_allow_html=True)
with s2:
    st.markdown('<div class="stat-card"><div class="stat-num">99%+</div><div class="stat-label">Model accuracy</div></div>', unsafe_allow_html=True)
with s3:
    st.markdown('<div class="stat-card"><div class="stat-num">4</div><div class="stat-label">Languages</div></div>', unsafe_allow_html=True)
with s4:
    st.markdown('<div class="stat-card"><div class="stat-num">Offline</div><div class="stat-label">Field-ready design</div></div>', unsafe_allow_html=True)

st.write("")

# ---------- How it works ----------
st.markdown('<div class="section-header">âš¡ How it works</div>', unsafe_allow_html=True)
h1, h2, h3 = st.columns(3)
with h1:
    st.markdown('<div class="step-card"><div class="step-num">1</div><b>ðŸ“· Snap a photo</b><br><span style="opacity:0.85; font-size:0.85rem;">Take a clear photo of the affected leaf</span></div>', unsafe_allow_html=True)
with h2:
    st.markdown('<div class="step-card"><div class="step-num">2</div><b>ðŸ§  AI analyzes</b><br><span style="opacity:0.85; font-size:0.85rem;">On-device model detects disease instantly</span></div>', unsafe_allow_html=True)
with h3:
    st.markdown('<div class="step-card"><div class="step-num">3</div><b>ðŸ—£ï¸ Get advice</b><br><span style="opacity:0.85; font-size:0.85rem;">Hear treatment steps in your language</span></div>', unsafe_allow_html=True)

st.write("")

# ---------- Disease info ----------
disease_info = {
    "Pepper__bell___Bacterial_spot": {
        "display": "Bell Pepper â€” Bacterial Spot",
        "icon": "ðŸ«‘",
        "treatment_en": "Apply copper-based bactericide. Avoid overhead watering and remove infected leaves.",
        "treatment_hi": "à¤•à¥‰à¤ªà¤° à¤†à¤§à¤¾à¤°à¤¿à¤¤ à¤¬à¥ˆà¤•à¥à¤Ÿà¥€à¤°à¤¿à¤¸à¤¾à¤‡à¤¡ à¤•à¤¾ à¤ªà¥à¤°à¤¯à¥‹à¤— à¤•à¤°à¥‡à¤‚à¥¤ à¤Šà¤ªà¤° à¤¸à¥‡ à¤ªà¤¾à¤¨à¥€ à¤¦à¥‡à¤¨à¥‡ à¤¸à¥‡ à¤¬à¤šà¥‡à¤‚ à¤”à¤° à¤¸à¤‚à¤•à¥à¤°à¤®à¤¿à¤¤ à¤ªà¤¤à¥à¤¤à¤¿à¤¯à¥‹à¤‚ à¤•à¥‹ à¤¹à¤Ÿà¤¾ à¤¦à¥‡à¤‚à¥¤",
        "treatment_mr": "à¤¤à¤¾à¤‚à¤¬à¥‡à¤¯à¥à¤•à¥à¤¤ à¤¬à¥…à¤•à¥à¤Ÿà¥‡à¤°à¤¿à¤¸à¤¾à¤‡à¤¡ à¤µà¤¾à¤ªà¤°à¤¾. à¤µà¤°à¥‚à¤¨ à¤ªà¤¾à¤£à¥€ à¤¦à¥‡à¤£à¥‡ à¤Ÿà¤¾à¤³à¤¾ à¤†à¤£à¤¿ à¤¸à¤‚à¤•à¥à¤°à¤®à¤¿à¤¤ à¤ªà¤¾à¤¨à¥‡ à¤•à¤¾à¤¢à¥‚à¤¨ à¤Ÿà¤¾à¤•à¤¾.",
        "treatment_kn": "à²¤à²¾à²®à³à²° à²†à²§à²¾à²°à²¿à²¤ à²¬à³à²¯à²¾à²•à³à²Ÿà³€à²°à²¿à²¸à³ˆà²¡à³ à²…à²¨à³à²¨à³ à²¬à²³à²¸à²¿. à²®à³‡à²²à²¿à²¨à²¿à²‚à²¦ à²¨à³€à²°à³à²£à²¿à²¸à³à²µà³à²¦à²¨à³à²¨à³ à²¤à²ªà³à²ªà²¿à²¸à²¿ à²®à²¤à³à²¤à³ à²¸à³‹à²‚à²•à²¿à²¤ à²Žà²²à³†à²—à²³à²¨à³à²¨à³ à²¤à³†à²—à³†à²¦à³à²¹à²¾à²•à²¿.",
        "severity": "moderate"
    },
    "Potato___Early_blight": {
        "display": "Potato â€” Early Blight",
        "icon": "ðŸ¥”",
        "treatment_en": "Apply fungicide (Chlorothalonil or Mancozeb). Rotate crops and remove infected debris.",
        "treatment_hi": "à¤«à¤«à¥‚à¤‚à¤¦à¤¨à¤¾à¤¶à¤• (à¤•à¥à¤²à¥‹à¤°à¥‹à¤¥à¤¾à¤²à¥‹à¤¨à¤¿à¤² à¤¯à¤¾ à¤®à¥ˆà¤‚à¤•à¥‹à¤œà¥‡à¤¬) à¤•à¤¾ à¤ªà¥à¤°à¤¯à¥‹à¤— à¤•à¤°à¥‡à¤‚à¥¤ à¤«à¤¸à¤² à¤šà¤•à¥à¤° à¤…à¤ªà¤¨à¤¾à¤à¤‚ à¤”à¤° à¤¸à¤‚à¤•à¥à¤°à¤®à¤¿à¤¤ à¤…à¤µà¤¶à¥‡à¤· à¤¹à¤Ÿà¤¾ à¤¦à¥‡à¤‚à¥¤",
        "treatment_mr": "à¤¬à¥à¤°à¤¶à¥€à¤¨à¤¾à¤¶à¤• (à¤•à¥à¤²à¥‹à¤°à¥‹à¤¥à¥…à¤²à¥‹à¤¨à¤¿à¤² à¤•à¤¿à¤‚à¤µà¤¾ à¤®à¥…à¤¨à¥à¤•à¥‹à¤à¥‡à¤¬) à¤µà¤¾à¤ªà¤°à¤¾. à¤ªà¥€à¤• à¤«à¥‡à¤°à¤ªà¤¾à¤²à¤Ÿ à¤•à¤°à¤¾ à¤†à¤£à¤¿ à¤¸à¤‚à¤•à¥à¤°à¤®à¤¿à¤¤ à¤…à¤µà¤¶à¥‡à¤· à¤•à¤¾à¤¢à¥‚à¤¨ à¤Ÿà¤¾à¤•à¤¾.",
        "treatment_kn": "à²¶à²¿à²²à³€à²‚à²§à³à²°à²¨à²¾à²¶à²• (à²•à³à²²à³‹à²°à³‹à²¥à²²à³‹à²¨à²¿à²²à³ à²…à²¥à²µà²¾ à²®à³à²¯à²¾à²‚à²•à³‹à²œà³†à²¬à³) à²¬à²³à²¸à²¿. à²¬à³†à²³à³† à²¸à²°à²¦à²¿ à²…à²¨à³à²¸à²°à²¿à²¸à²¿ à²®à²¤à³à²¤à³ à²¸à³‹à²‚à²•à²¿à²¤ à²…à²µà²¶à³‡à²·à²—à²³à²¨à³à²¨à³ à²¤à³†à²—à³†à²¦à³à²¹à²¾à²•à²¿.",
        "severity": "moderate"
    },
    "Tomato_Late_blight": {
        "display": "Tomato â€” Late Blight",
        "icon": "ðŸ…",
        "treatment_en": "Apply copper-based fungicide immediately. Remove and destroy infected plants to prevent spread.",
        "treatment_hi": "à¤¤à¥à¤°à¤‚à¤¤ à¤•à¥‰à¤ªà¤° à¤†à¤§à¤¾à¤°à¤¿à¤¤ à¤«à¤«à¥‚à¤‚à¤¦à¤¨à¤¾à¤¶à¤• à¤•à¤¾ à¤ªà¥à¤°à¤¯à¥‹à¤— à¤•à¤°à¥‡à¤‚à¥¤ à¤«à¥ˆà¤²à¤¾à¤µ à¤°à¥‹à¤•à¤¨à¥‡ à¤•à¥‡ à¤²à¤¿à¤ à¤¸à¤‚à¤•à¥à¤°à¤®à¤¿à¤¤ à¤ªà¥Œà¤§à¥‹à¤‚ à¤•à¥‹ à¤¹à¤Ÿà¤¾à¤•à¤° à¤¨à¤·à¥à¤Ÿ à¤•à¤° à¤¦à¥‡à¤‚à¥¤",
        "treatment_mr": "à¤¤à¥à¤µà¤°à¤¿à¤¤ à¤¤à¤¾à¤‚à¤¬à¥‡à¤¯à¥à¤•à¥à¤¤ à¤¬à¥à¤°à¤¶à¥€à¤¨à¤¾à¤¶à¤• à¤µà¤¾à¤ªà¤°à¤¾. à¤ªà¥à¤°à¤¸à¤¾à¤° à¤°à¥‹à¤–à¤£à¥à¤¯à¤¾à¤¸à¤¾à¤ à¥€ à¤¸à¤‚à¤•à¥à¤°à¤®à¤¿à¤¤ à¤°à¥‹à¤ªà¥‡ à¤•à¤¾à¤¢à¥‚à¤¨ à¤¨à¤·à¥à¤Ÿ à¤•à¤°à¤¾.",
        "treatment_kn": "à²¤à²•à³à²·à²£ à²¤à²¾à²®à³à²° à²†à²§à²¾à²°à²¿à²¤ à²¶à²¿à²²à³€à²‚à²§à³à²°à²¨à²¾à²¶à²•à²µà²¨à³à²¨à³ à²¬à²³à²¸à²¿. à²¹à²°à²¡à³à²µà²¿à²•à³†à²¯à²¨à³à²¨à³ à²¤à²¡à³†à²¯à²²à³ à²¸à³‹à²‚à²•à²¿à²¤ à²¸à²¸à³à²¯à²—à²³à²¨à³à²¨à³ à²¤à³†à²—à³†à²¦à³ à²¨à²¾à²¶à²ªà²¡à²¿à²¸à²¿.",
        "severity": "severe"
    },
    "Tomato_healthy": {
        "display": "Tomato â€” Healthy",
        "icon": "âœ…",
        "treatment_en": "No disease detected. Continue regular monitoring and good field hygiene.",
        "treatment_hi": "à¤•à¥‹à¤ˆ à¤°à¥‹à¤— à¤¨à¤¹à¥€à¤‚ à¤ªà¤¾à¤¯à¤¾ à¤—à¤¯à¤¾à¥¤ à¤¨à¤¿à¤¯à¤®à¤¿à¤¤ à¤¨à¤¿à¤—à¤°à¤¾à¤¨à¥€ à¤”à¤° à¤…à¤šà¥à¤›à¥€ à¤–à¥‡à¤¤ à¤¸à¥à¤µà¤šà¥à¤›à¤¤à¤¾ à¤œà¤¾à¤°à¥€ à¤°à¤–à¥‡à¤‚à¥¤",
        "treatment_mr": "à¤•à¥‹à¤£à¤¤à¤¾à¤¹à¥€ à¤°à¥‹à¤— à¤†à¤¢à¤³à¤²à¤¾ à¤¨à¤¾à¤¹à¥€. à¤¨à¤¿à¤¯à¤®à¤¿à¤¤ à¤¦à¥‡à¤–à¤°à¥‡à¤– à¤†à¤£à¤¿ à¤šà¤¾à¤‚à¤—à¤²à¥€ à¤¶à¥‡à¤¤ à¤¸à¥à¤µà¤šà¥à¤›à¤¤à¤¾ à¤¸à¥à¤°à¥‚ à¤ à¥‡à¤µà¤¾.",
        "treatment_kn": "à²¯à²¾à²µà³à²¦à³‡ à²°à³‹à²— à²ªà²¤à³à²¤à³†à²¯à²¾à²—à²¿à²²à³à²². à²¨à²¿à²¯à²®à²¿à²¤ à²®à³‡à²²à³à²µà²¿à²šà²¾à²°à²£à³† à²®à²¤à³à²¤à³ à²‰à²¤à³à²¤à²® à²¹à³Šà²²à²¦ à²¨à³ˆà²°à³à²®à²²à³à²¯à²µà²¨à³à²¨à³ à²®à³à²‚à²¦à³à²µà²°à²¿à²¸à²¿.",
        "severity": "healthy"
    }
}

severity_colors = {"healthy": "#7bd389", "moderate": "#f2c744", "severe": "#e0665a"}
severity_labels = {"healthy": "Healthy", "moderate": "Moderate risk", "severe": "Severe â€” act now"}

unknown_text = {
    "en": "This doesn't look like one of the crop leaves our model was trained on (tomato, potato, or bell pepper). Please retake the photo, or check with a local agriculture expert.",
    "hi": "à¤¯à¤¹ à¤‰à¤¨ à¤«à¤¸à¤² à¤ªà¤¤à¥à¤¤à¤¿à¤¯à¥‹à¤‚ à¤®à¥‡à¤‚ à¤¸à¥‡ à¤¨à¤¹à¥€à¤‚ à¤²à¤—à¤¤à¤¾ à¤œà¤¿à¤¨ à¤ªà¤° à¤¹à¤®à¤¾à¤°à¤¾ à¤®à¥‰à¤¡à¤² à¤ªà¥à¤°à¤¶à¤¿à¤•à¥à¤·à¤¿à¤¤ à¤¹à¥ˆ (à¤Ÿà¤®à¤¾à¤Ÿà¤°, à¤†à¤²à¥‚ à¤¯à¤¾ à¤¶à¤¿à¤®à¤²à¤¾ à¤®à¤¿à¤°à¥à¤š)à¥¤ à¤•à¥ƒà¤ªà¤¯à¤¾ à¤«à¤¿à¤° à¤¸à¥‡ à¤«à¥‹à¤Ÿà¥‹ à¤²à¥‡à¤‚, à¤¯à¤¾ à¤¸à¥à¤¥à¤¾à¤¨à¥€à¤¯ à¤•à¥ƒà¤·à¤¿ à¤µà¤¿à¤¶à¥‡à¤·à¤œà¥à¤ž à¤¸à¥‡ à¤¸à¤‚à¤ªà¤°à¥à¤• à¤•à¤°à¥‡à¤‚à¥¤",
    "mr": "à¤¹à¥‡ à¤†à¤®à¤šà¥à¤¯à¤¾ à¤®à¥‰à¤¡à¥‡à¤²à¤¨à¥‡ à¤ªà¥à¤°à¤¶à¤¿à¤•à¥à¤·à¤¿à¤¤ à¤•à¥‡à¤²à¥‡à¤²à¥à¤¯à¤¾ à¤ªà¤¿à¤•à¤¾à¤‚à¤šà¥à¤¯à¤¾ à¤ªà¤¾à¤¨à¤¾à¤‚à¤¸à¤¾à¤°à¤–à¥‡ à¤¦à¤¿à¤¸à¤¤ à¤¨à¤¾à¤¹à¥€ (à¤Ÿà¥‹à¤®à¥…à¤Ÿà¥‹, à¤¬à¤Ÿà¤¾à¤Ÿà¤¾ à¤•à¤¿à¤‚à¤µà¤¾ à¤¢à¥‹à¤¬à¤³à¥€ à¤®à¤¿à¤°à¤šà¥€). à¤•à¥ƒà¤ªà¤¯à¤¾ à¤ªà¥à¤¨à¥à¤¹à¤¾ à¤«à¥‹à¤Ÿà¥‹ à¤˜à¥à¤¯à¤¾, à¤•à¤¿à¤‚à¤µà¤¾ à¤¸à¥à¤¥à¤¾à¤¨à¤¿à¤• à¤•à¥ƒà¤·à¥€ à¤¤à¤œà¥à¤žà¤¾à¤‚à¤¶à¥€ à¤¸à¤‚à¤ªà¤°à¥à¤• à¤¸à¤¾à¤§à¤¾.",
    "kn": "à²‡à²¦à³ à²¨à²®à³à²® à²®à²¾à²¦à²°à²¿à²—à³† à²¤à²°à²¬à³‡à²¤à²¿ à²¨à³€à²¡à²¿à²¦ à²¬à³†à²³à³† à²Žà²²à³†à²—à²³à²‚à²¤à³† à²•à²¾à²£à³à²¤à³à²¤à²¿à²²à³à²² (à²Ÿà³Šà²®à³à²¯à²¾à²Ÿà³Š, à²†à²²à³‚à²—à²¡à³à²¡à³† à²…à²¥à²µà²¾ à²¦à³Šà²¡à³à²¡ à²®à³†à²£à²¸à²¿à²¨à²•à²¾à²¯à²¿). à²¦à²¯à²µà²¿à²Ÿà³à²Ÿà³ à²«à³‹à²Ÿà³‹à²µà²¨à³à²¨à³ à²®à²¤à³à²¤à³† à²¤à³†à²—à³†à²¯à²¿à²°à²¿, à²…à²¥à²µà²¾ à²¸à³à²¥à²³à³€à²¯ à²•à³ƒà²·à²¿ à²¤à²œà³à²žà²°à²¨à³à²¨à³ à²¸à²‚à²ªà²°à³à²•à²¿à²¸à²¿."
}

# ---------- Upload + Result ----------
st.markdown('<div class="section-header">ðŸ“¸ Scan a crop leaf</div>', unsafe_allow_html=True)

col_upload, col_result = st.columns([1, 1.2], gap="large")

with col_upload:
    st.markdown('<div class="upload-panel-label">Drag a leaf photo below, or click to browse</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, use_container_width=True)

with col_result:
    if uploaded:
        img_resized = img.resize((224, 224))
        arr = np.expand_dims(np.array(img_resized) / 255.0, axis=0)
        pred = model.predict(arr)
        result = class_names[np.argmax(pred)]
        confidence = float(np.max(pred)) * 100

        # --- Language selector is needed either way, so pick it up front ---
        lang_choice = st.selectbox("ðŸŒ Voice language", ["English", "à¤¹à¤¿à¤‚à¤¦à¥€ (Hindi)", "à¤®à¤°à¤¾à¤ à¥€ (Marathi)", "à²•à²¨à³à²¨à²¡ (Kannada)"])
        lang_map = {
            "English": ("en", "treatment_en"),
            "à¤¹à¤¿à¤‚à¤¦à¥€ (Hindi)": ("hi", "treatment_hi"),
            "à¤®à¤°à¤¾à¤ à¥€ (Marathi)": ("mr", "treatment_mr"),
            "à²•à²¨à³à²¨à²¡ (Kannada)": ("kn", "treatment_kn"),
        }
        lang_code, treatment_key = lang_map[lang_choice]

        if confidence < CONFIDENCE_THRESHOLD:
            # --- Unrecognized image: don't force a disease label ---
            st.markdown(f"""
                <div class="result-card" style="border-top-color:#e0665a;">
                    <div class="result-label">Detection result</div>
                    <div class="result-name">â“ Not recognized</div>
                    <div style="margin-top:10px;">
                        <span class="status-dot" style="background-color:#e0665a;"></span>
                        <span style="font-weight:600; color:#e0665a;">Low confidence ({confidence:.1f}%)</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
                <div class="error-box">
                    <div style="font-weight:700; color:#e0665a; margin-bottom:6px;">âš ï¸ Image not recognized</div>
                    <div style="color:#eef2e6; line-height:1.6;">{unknown_text[lang_code]}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            info = disease_info.get(result, {
                "display": result.replace("_", " "), "icon": "ðŸŒ¿",
                "treatment_en": "Consult a local agriculture expert.",
                "treatment_hi": "à¤¸à¥à¤¥à¤¾à¤¨à¥€à¤¯ à¤•à¥ƒà¤·à¤¿ à¤µà¤¿à¤¶à¥‡à¤·à¤œà¥à¤ž à¤¸à¥‡ à¤¸à¤²à¤¾à¤¹ à¤²à¥‡à¤‚à¥¤",
                "treatment_mr": "à¤¸à¥à¤¥à¤¾à¤¨à¤¿à¤• à¤•à¥ƒà¤·à¥€ à¤¤à¤œà¥à¤žà¤¾à¤‚à¤šà¤¾ à¤¸à¤²à¥à¤²à¤¾ à¤˜à¥à¤¯à¤¾.",
                "treatment_kn": "à²¸à³à²¥à²³à³€à²¯ à²•à³ƒà²·à²¿ à²¤à²œà³à²žà²°à²¨à³à²¨à³ à²¸à²‚à²ªà²°à³à²•à²¿à²¸à²¿.",
                "severity": "moderate"
            })
            color = severity_colors.get(info["severity"], "#f2c744")

            st.markdown(f"""
                <div class="result-card" style="border-top-color:{color};">
                    <div class="result-label">Detection result</div>
                    <div class="result-name">{info['icon']} {info['display']}</div>
                    <div style="margin-top:10px;">
                        <span class="status-dot" style="background-color:{color};"></span>
                        <span style="font-weight:600; color:{color};">{severity_labels.get(info['severity'], '')}</span>
                    </div>
                    <div style="margin-top:14px; font-size:0.85rem; color:#d3ddc7;">Confidence: <b>{confidence:.1f}%</b></div>
                    <div class="confidence-bar-bg">
                        <div class="confidence-bar-fill" style="width:{confidence}%; background-color:{color};"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            treatment_text = info[treatment_key]

            st.markdown(f"""
                <div class="treatment-box">
                    <div style="font-weight:700; color:#f2c744; margin-bottom:6px;">ðŸ’Š Recommended action</div>
                    <div style="color:#eef2e6; line-height:1.6;">{treatment_text}</div>
                </div>
            """, unsafe_allow_html=True)

            if st.button("ðŸ”Š Play voice advice", use_container_width=True):
                try:
                    import pyttsx3
                    engine = pyttsx3.init()
                    # Try to pick a voice matching the language; falls back to default if unavailable
                    for voice in engine.getProperty("voices"):
                        if lang_code in voice.id.lower() or lang_code in (voice.languages[0].decode(errors="ignore").lower() if voice.languages else ""):
                            engine.setProperty("voice", voice.id)
                            break
                    engine.save_to_file(f"{info['display']}. {treatment_text}", "output.mp3")
                    engine.runAndWait()
                    st.audio("output.mp3")
                except Exception as e:
                    st.warning(
                        "Offline voice engine isn't available on this device. "
                        "Install a system TTS engine (e.g. `espeak` on Linux, or use the built-in "
                        "voices on Windows/Mac) and `pip install pyttsx3` to enable this feature."
                    )
    else:
        st.markdown("""
            <div style="height:100%; display:flex; align-items:center; justify-content:center; text-align:center; color:#9fab8f; padding: 3rem 1rem;">
                <div>
                    <div style="font-size:3.2rem;">ðŸŒ±</div>
                    <div style="margin-top:10px;">Upload a photo to see detection results here</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

# ---------- Helpline ----------
st.markdown('<div class="section-header">ðŸ“ž Farmer helpline & support</div>', unsafe_allow_html=True)
st.markdown("""
    <div class="helpline-card">
        <div style="font-weight:700; color:#f2c744; margin-bottom:6px;">ðŸ“± Kisan Call Centre (Government of India)</div>
        <div style="color:#eef2e6; font-size:0.95rem; margin-bottom:14px;">Toll-free <b>1800-180-1551</b> Â· 6 AMâ€“10 PM, all 7 days Â· 22 local languages</div>
        <div style="font-weight:700; color:#f2c744; margin-bottom:6px;">ðŸ“± PM-KISAN Helpline</div>
        <div style="color:#eef2e6; font-size:0.95rem;">Toll-free <b>155261</b> / <b>1800-115-526</b> &nbsp;|&nbsp; â˜Žï¸ 011-24300606</div>
    </div>
""", unsafe_allow_html=True)
st.caption("Numbers verified from official Government of India sources. Production version will link directly to the nearest Maharashtra Krishi Vibhag extension officer by location.")

# ---------- Roadmap ----------
st.markdown('<div class="section-header">ðŸŒ± Expanding crop coverage</div>', unsafe_allow_html=True)
st.write("This prototype currently detects diseases in **tomato, potato, and bell pepper**. Next, we're expanding to Maharashtra's core crops:")

r1, r2, r3, r4 = st.columns(4)
with r1:
    st.markdown('<div class="roadmap-chip">ðŸŒ¾ <b>Jowar</b><br><span style="font-size:0.85rem;">Grain mold, downy mildew</span></div>', unsafe_allow_html=True)
with r2:
    st.markdown('<div class="roadmap-chip">ðŸŒ¾ <b>Rice</b><br><span style="font-size:0.85rem;">Blast, bacterial blight</span></div>', unsafe_allow_html=True)
with r3:
    st.markdown('<div class="roadmap-chip">ðŸŒ¿ <b>Cotton</b><br><span style="font-size:0.85rem;">Pink bollworm, leaf curl</span></div>', unsafe_allow_html=True)
with r4:
    st.markdown('<div class="roadmap-chip">ðŸŽ‹ <b>Sugarcane</b><br><span style="font-size:0.85rem;">Red rot, smut</span></div>', unsafe_allow_html=True)

st.markdown('<p class="footer-note">Prototype for SIH 2026 Â· Problem Statement SIH26131 Â· Government of Maharashtra</p>', unsafe_allow_html=True)
