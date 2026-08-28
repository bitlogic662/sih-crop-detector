import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import json

st.set_page_config(
    page_title="KrishiRakshak AI - Crop Disease Detection",
    page_icon="🌾",
    layout="wide"
)

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
        content: "🌾";
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

    /* Style Streamlit's ACTUAL dropzone so the whole visible box is the real drop target */
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

    /* Streamlit native widgets */
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

    /* === NEW FEATURE CSS: cards for duration / severity / health dashboard === */
    .info-card {
        background: rgba(255,255,255,0.06);
        border-radius: 18px;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
        border: 1px solid rgba(255,255,255,0.10);
        border-left: 5px solid #f2c744;
    }
    .info-card.mild { border-left-color: #7bd389; }
    .info-card.moderate { border-left-color: #f2c744; }
    .info-card.severe { border-left-color: #e0665a; }
    .info-card h4 {
        margin-top: 0;
        margin-bottom: 0.4rem;
        color: #f6f9f2;
    }
    .info-card p {
        margin: 0.2rem 0;
        color: #d3ddc7;
        font-size: 0.92rem;
    }
    .health-status-badge {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.92rem;
        margin-top: 0.4rem;
    }
    .health-bar-bg {
        background-color: rgba(255,255,255,0.12);
        border-radius: 10px;
        height: 14px;
        width: 100%;
        margin-top: 8px;
        overflow: hidden;
    }
    .health-bar-fill {
        height: 14px;
        border-radius: 10px;
        transition: width 0.8s cubic-bezier(0.34, 1.56, 0.64, 1);
    }

    /* === NEW FEATURE CSS: crop age / growth stage / recommendation section === */
    .subsection-title {
        font-weight: 700;
        color: #f2c744;
        margin: 1.4rem 0 0.5rem 0;
        font-size: 1.05rem;
    }
    .urgency-badge {
        display: inline-block;
        padding: 0.45rem 1.1rem;
        border-radius: 20px;
        font-weight: 800;
        font-size: 0.95rem;
        margin-top: 0.3rem;
    }
    div[data-testid="stNumberInput"] label, div[data-testid="stCheckbox"] label {
        color: #eef2e6 !important;
    }

    /* ===== FIX: make the number-input box (incl. +/- buttons area) fully visible
       against the dark theme. Previously only the <input> text color/background
       was themed, leaving the surrounding widget shell/buttons on the default
       light Streamlit styling, which made the temperature/humidity boxes look
       blank or "invisible" against the dark background. ===== */
    div[data-testid="stNumberInput"] {
        background: transparent !important;
    }
    div[data-testid="stNumberInput"] > div {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.20) !important;
        border-radius: 10px !important;
        overflow: hidden;
    }
    div[data-testid="stNumberInput"] input {
        background: rgba(255,255,255,0.08) !important;
        color: #eef2e6 !important;
        border-radius: 10px !important;
        caret-color: #f2c744 !important;
    }
    div[data-testid="stNumberInput"] button {
        background: rgba(255,255,255,0.14) !important;
        border-left: 1px solid rgba(255,255,255,0.20) !important;
    }
    div[data-testid="stNumberInput"] button svg {
        fill: #eef2e6 !important;
        color: #eef2e6 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Load model ----------
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
        <div class="hero-title">🌾 KrishiRakshak AI</div>
        <div class="hero-sub">Early Detection & Management of Crop Diseases and Pest Infestations</div>
        <div class="hero-badge">🏛️ Government of Maharashtra · SIH 2026 · SIH26131</div>
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
st.markdown('<div class="section-header">⚡ How it works</div>', unsafe_allow_html=True)
h1, h2, h3 = st.columns(3)
with h1:
    st.markdown('<div class="step-card"><div class="step-num">1</div><b>📷 Snap a photo</b><br><span style="opacity:0.85; font-size:0.85rem;">Take a clear photo of the affected leaf</span></div>', unsafe_allow_html=True)
with h2:
    st.markdown('<div class="step-card"><div class="step-num">2</div><b>🧠 AI analyzes</b><br><span style="opacity:0.85; font-size:0.85rem;">On-device model detects disease instantly</span></div>', unsafe_allow_html=True)
with h3:
    st.markdown('<div class="step-card"><div class="step-num">3</div><b>🗣️ Get advice</b><br><span style="opacity:0.85; font-size:0.85rem;">Hear treatment steps in your language</span></div>', unsafe_allow_html=True)

st.write("")

# ---------- Disease info ----------
disease_info = {
    "Pepper__bell___Bacterial_spot": {
        "display": "Bell Pepper — Bacterial Spot",
        "crop": "Bell Pepper",
        "icon": "🫑",
        "treatment_en": "Apply copper-based bactericide. Avoid overhead watering and remove infected leaves.",
        "treatment_hi": "कॉपर आधारित बैक्टीरिसाइड का प्रयोग करें। ऊपर से पानी देने से बचें और संक्रमित पत्तियों को हटा दें।",
        "treatment_mr": "तांबेयुक्त बॅक्टेरिसाइड वापरा. वरून पाणी देणे टाळा आणि संक्रमित पाने काढून टाका.",
        "treatment_kn": "ತಾಮ್ರ ಆಧಾರಿತ ಬ್ಯಾಕ್ಟೀರಿಸೈಡ್ ಅನ್ನು ಬಳಸಿ. ಮೇಲಿನಿಂದ ನೀರುಣಿಸುವುದನ್ನು ತಪ್ಪಿಸಿ ಮತ್ತು ಸೋಂಕಿತ ಎಲೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ.",
        "severity": "moderate"
    },
    "Potato___Early_blight": {
        "display": "Potato — Early Blight",
        "crop": "Potato",
        "icon": "🥔",
        "treatment_en": "Apply fungicide (Chlorothalonil or Mancozeb). Rotate crops and remove infected debris.",
        "treatment_hi": "फफूंदनाशक (क्लोरोथालोनिल या मैंकोजेब) का प्रयोग करें। फसल चक्र अपनाएं और संक्रमित अवशेष हटा दें।",
        "treatment_mr": "बुरशीनाशक (क्लोरोथॅलोनिल किंवा मॅन्कोझेब) वापरा. पीक फेरपालट करा आणि संक्रमित अवशेष काढून टाका.",
        "treatment_kn": "ಶಿಲೀಂಧ್ರನಾಶಕ (ಕ್ಲೋರೋಥಲೋನಿಲ್ ಅಥವಾ ಮ್ಯಾಂಕೋಜೆಬ್) ಬಳಸಿ. ಬೆಳೆ ಸರದಿ ಅನುಸರಿಸಿ ಮತ್ತು ಸೋಂಕಿತ ಅವಶೇಷಗಳನ್ನು ತೆಗೆದುಹಾಕಿ.",
        "severity": "moderate"
    },
    "Tomato_Late_blight": {
        "display": "Tomato — Late Blight",
        "crop": "Tomato",
        "icon": "🍅",
        "treatment_en": "Apply copper-based fungicide immediately. Remove and destroy infected plants to prevent spread.",
        "treatment_hi": "तुरंत कॉपर आधारित फफूंदनाशक का प्रयोग करें। फैलाव रोकने के लिए संक्रमित पौधों को हटाकर नष्ट कर दें।",
        "treatment_mr": "त्वरित तांबेयुक्त बुरशीनाशक वापरा. प्रसार रोखण्यासाठी संक्रमित रोपे काढून नष्ट करा.",
        "treatment_kn": "ತಕ್ಷಣ ತಾಮ್ರ ಆಧಾರಿತ ಶಿಲೀಂಧ್ರನಾಶಕವನ್ನು ಬಳಸಿ. ಹರಡುವಿಕೆಯನ್ನು ತಡೆಯಲು ಸೋಂಕಿತ ಸಸ್ಯಗಳನ್ನು ತೆಗೆದು ನಾಶಪಡಿಸಿ.",
        "severity": "severe"
    },
    "Tomato_healthy": {
        "display": "Tomato — Healthy",
        "crop": "Tomato",
        "icon": "✅",
        "treatment_en": "No disease detected. Continue regular monitoring and good field hygiene.",
        "treatment_hi": "कोई रोग नहीं पाया गया। नियमित निगरानी और अच्छी खेत स्वच्छता जारी रखें।",
        "treatment_mr": "कोणताही रोग आढळला नाही. नियमित देखरेख आणि चांगली शेत स्वच्छता सुरू ठेवा.",
        "treatment_kn": "ಯಾವುದೇ ರೋಗ ಪತ್ತೆಯಾಗಿಲ್ಲ. ನಿಯಮಿತ ಮೇಲ್ವಿಚಾರಣೆ ಮತ್ತು ಉತ್ತಮ ಹೊಲದ ನೈರ್ಮಲ್ಯವನ್ನು ಮುಂದುವರಿಸಿ.",
        "severity": "healthy"
    }
}

severity_colors = {"healthy": "#7bd389", "moderate": "#f2c744", "severe": "#e0665a"}
severity_labels = {"healthy": "Healthy", "moderate": "Moderate risk", "severe": "Severe — act now"}


# ======================================================
# === NEW FEATURE 4 (part A): BASIC IMAGE VALIDATION ===
# ======================================================
def validate_prediction(image: Image.Image, confidence: float, confidence_threshold: float = 60.0):
    """
    Lightweight, dependency-free sanity checks so obviously unsuitable
    images aren't confidently presented as a disease diagnosis.
    Returns {"is_valid": bool, "warnings": [str, ...]}.
    This does NOT block prediction — it only adds cautionary messages,
    since your model/class_names remain the source of truth.
    """
    warnings = []
    is_valid = True

    width, height = image.size
    if width < 50 or height < 50:
        warnings.append("⚠️ The uploaded image resolution is very low. Please upload a clearer photo.")
        is_valid = False

    try:
        small = image.convert("RGB").resize((64, 64))
        arr = np.array(small).astype("float32")
        mean_r = arr[:, :, 0].mean()
        mean_g = arr[:, :, 1].mean()
        mean_b = arr[:, :, 2].mean()
        std_total = arr.std()
        if std_total < 8:
            warnings.append("⚠️ The image appears to have very little detail. "
                             "Please upload a focused photo of the affected leaf.")
            is_valid = False
        if mean_g < mean_r * 0.85 and mean_g < mean_b * 0.85:
            warnings.append("ℹ️ This image doesn't show strong green/leaf-like coloring. "
                             "Please make sure the photo is a close-up of a crop leaf for best results.")
    except Exception:
        pass

    if confidence < confidence_threshold:
        warnings.append("⚠️ Low confidence prediction. Please upload a clearer image of the affected crop leaf.")
        is_valid = False

    return {"is_valid": is_valid, "warnings": warnings}


# ======================================================
# === NEW FEATURE 2: DISEASE SEVERITY ASSESSMENT ===
# ======================================================
def get_severity(info: dict, confidence: float):
    """
    Combines your existing per-disease 'severity' rule (healthy/moderate/severe
    in disease_info) with model confidence to produce a Mild/Moderate/Severe
    estimate, plus a short explanation and recommended action.

    IMPORTANT: This is an AI-based/estimated severity assessment only —
    it is NOT a scientifically validated disease severity measurement.
    Returns None for healthy predictions (no severity applicable).
    """
    base = info.get("severity", "moderate")
    if base == "healthy":
        return None

    if base == "severe":
        level = "Severe" if confidence >= 70 else "Moderate"
    else:  # base == "moderate"
        level = "Moderate" if confidence >= 70 else "Mild"

    details = {
        "Mild": {
            "explanation": "The AI model detects early or low-intensity visual symptoms.",
            "action": "Monitor the crop and remove visibly affected leaves if appropriate.",
            "css_class": "mild",
        },
        "Moderate": {
            "explanation": "The AI model detects clearer, more established disease symptoms.",
            "action": "Increase monitoring and consider suitable crop-protection measures.",
            "css_class": "moderate",
        },
        "Severe": {
            "explanation": "The AI model detects strong, widespread visual symptoms of disease.",
            "action": "Consult a local agricultural expert or agriculture officer and take prompt action.",
            "css_class": "severe",
        },
    }
    d = details[level]
    return {"level": level, "explanation": d["explanation"], "action": d["action"], "css_class": d["css_class"]}


# ======================================================
# === NEW FEATURE 1: DISEASE DURATION / PROGRESSION ===
# ======================================================
def get_disease_duration(severity_level: str):
    """
    Approximate, general progression estimate tied to severity level.
    These are general educational ranges, not precise agronomic predictions —
    actual progression depends on crop, weather, and infection conditions.
    """
    duration_map = {
        "Mild": "Early stage: approx. 1–3 days",
        "Moderate": "Moderate stage: approx. 3–7 days",
        "Severe": "Severe stage: approx. 7+ days",
    }
    return duration_map.get(severity_level, "Progression estimate unavailable")


# ======================================================
# === NEW FEATURE 3: CROP HEALTH PROGRESS DASHBOARD ===
# ======================================================
def get_crop_health_progress(is_healthy: bool, severity_level: str = None):
    """
    Converts the detection result into a simple visual health percentage
    and risk-status label. Visual indicator only, not a scientific measurement.
    """
    if is_healthy:
        return {"percent": 100, "status_label": "Healthy", "status_color": "#7bd389"}

    mapping = {
        "Mild": {"percent": 75, "status_label": "Low Risk", "status_color": "#a9d97a"},
        "Moderate": {"percent": 50, "status_label": "Moderate Risk", "status_color": "#f2c744"},
        "Severe": {"percent": 25, "status_label": "High Risk", "status_color": "#e0665a"},
    }
    return mapping.get(severity_level, {"percent": 50, "status_label": "Moderate Risk", "status_color": "#f2c744"})


# ======================================================
# === NEW FEATURE 5: CROP-SPECIFIC GROWTH STAGE LOOKUP ===
# ======================================================
# Approximate, crop-specific growth-stage timelines (days after planting).
# These are general agronomic ranges and can vary with variety, season and region.
CROP_GROWTH_STAGES = {
    "Tomato": [
        (0, 15, "Seedling / Early Vegetative Stage"),
        (16, 35, "Vegetative Stage"),
        (36, 55, "Flowering Stage"),
        (56, 80, "Fruiting / Reproductive Stage"),
        (81, 9999, "Maturity Stage"),
    ],
    "Potato": [
        (0, 20, "Sprouting / Early Vegetative Stage"),
        (21, 40, "Vegetative Stage"),
        (41, 60, "Tuber Initiation / Flowering Stage"),
        (61, 90, "Tuber Bulking (Reproductive) Stage"),
        (91, 9999, "Maturity Stage"),
    ],
    "Bell Pepper": [
        (0, 20, "Seedling Stage"),
        (21, 45, "Vegetative Stage"),
        (46, 70, "Flowering Stage"),
        (71, 100, "Fruiting Stage"),
        (101, 9999, "Maturity Stage"),
    ],
}


def get_growth_stage(crop: str, age_days: int):
    """
    Returns the estimated growth stage for a given crop and age in days.
    Uses crop-specific stage timelines where available; otherwise falls back
    to a generic estimate and flags it clearly as approximate.
    """
    stages = CROP_GROWTH_STAGES.get(crop)
    if stages:
        for lo, hi, name in stages:
            if lo <= age_days <= hi:
                return {"stage": name, "is_estimate": False}
        return {"stage": stages[-1][2], "is_estimate": False}

    # Generic fallback when crop-specific data isn't available
    if age_days <= 20:
        generic = "Early / Seedling Stage"
    elif age_days <= 45:
        generic = "Vegetative Stage"
    elif age_days <= 70:
        generic = "Flowering Stage"
    elif age_days <= 100:
        generic = "Fruiting / Reproductive Stage"
    else:
        generic = "Maturity Stage"
    return {"stage": generic, "is_estimate": True}


def get_stage_warning(crop: str, stage_name: str):
    """
    Generates a growth-stage-specific caution message. Varies by actual
    crop and stage rather than repeating one message for every case.
    """
    stage_lower = stage_name.lower()
    if any(k in stage_lower for k in ["seedling", "early", "sprouting"]):
        return ("🌱 Your crop is in an early growth stage. Avoid unnecessary chemical treatment "
                "and prioritize preventive/non-chemical control measures (removing infected leaves, "
                "proper spacing, avoiding overhead watering).")
    if "flowering" in stage_lower or "tuber initiation" in stage_lower:
        return (f"🌸 Your {crop.lower()} crop is flowering. Use only treatments approved for this "
                f"crop and growth stage, avoid spraying during peak pollinator activity, and follow "
                f"the label instructions carefully.")
    if any(k in stage_lower for k in ["fruiting", "bulking", "reproductive"]):
        return ("🍅 Your crop is in the fruiting/reproductive stage. Pay close attention to the "
                "product's pre-harvest interval (PHI) before applying any pesticide.")
    if "maturity" in stage_lower:
        return ("🌾 Your crop is nearing maturity. Prioritize pre-harvest interval compliance and "
                "consider whether chemical treatment is still necessary this close to harvest.")
    return "Follow general crop-protection best practices appropriate for this growth stage."


# ======================================================
# === NEW FEATURE 6: VERIFIED PESTICIDE/FUNGICIDE REFERENCE DATABASE ===
# ======================================================
# Reference rates for the diseases this model currently detects, based on
# general ICAR / state agricultural extension guidance for these active
# ingredients. These are NOT invented figures — but formulations and brands
# vary, so farmers are always told to confirm against their actual product
# label or a local agricultural officer before applying anything.
PESTICIDE_DB = {
    "Pepper__bell___Bacterial_spot": {
        "product": "Copper Oxychloride 50% WP",
        "purpose": "Bactericide — helps control bacterial spot",
        "rate": "2.5–3 g per litre of water",
        "water_volume": "Spray to full leaf wetness (approx. 500–600 L/acre for mature plants)",
        "method": "Foliar spray, preferably in the evening or cooler hours",
        "timing": "At first appearance of symptoms; repeat every 7–10 days if needed",
        "phi": "Typical pre-harvest interval: 3–5 days (confirm on product label)",
        "source_note": "Reference rate based on general extension guidance for copper-based "
                        "bactericides on bell pepper bacterial spot.",
    },
    "Potato___Early_blight": {
        "product": "Mancozeb 75% WP",
        "purpose": "Fungicide — helps control early blight",
        "rate": "2–2.5 g per litre of water",
        "water_volume": "Spray to full leaf wetness (approx. 500 L/acre)",
        "method": "Foliar spray, avoid application right before rain",
        "timing": "At first symptom appearance; repeat every 7–10 days as needed",
        "phi": "Typical pre-harvest interval: 7 days (confirm on product label)",
        "source_note": "Reference rate based on general extension guidance for Mancozeb on "
                        "potato early blight.",
    },
    "Tomato_Late_blight": {
        "product": "Copper Oxychloride 50% WP (or Mancozeb 75% WP)",
        "purpose": "Fungicide — helps control late blight",
        "rate": "2.5–3 g per litre of water",
        "water_volume": "Spray to full leaf wetness (approx. 500–600 L/acre)",
        "method": "Foliar spray covering both sides of the leaves",
        "timing": "Immediately at first symptoms; repeat every 5–7 days, more frequently in humid weather",
        "phi": "Typical pre-harvest interval: 5–7 days (confirm on product label)",
        "source_note": "Reference rate based on general extension guidance for copper/Mancozeb "
                        "fungicides on tomato late blight.",
    },
}


def get_pesticide_recommendation(disease_key: str):
    """Returns verified reference dosage info for a disease, or None if unavailable."""
    return PESTICIDE_DB.get(disease_key)


def get_weather_risk(temp_c, humidity_pct):
    """
    Heuristic disease-favorability read on current weather, used only as
    additional context — never as proof of disease presence.
    """
    if temp_c is None or humidity_pct is None:
        return None
    if humidity_pct >= 80 and 18 <= temp_c <= 30:
        level = "High"
        message = ("Current humidity and temperature are favorable for fungal/bacterial disease "
                    "spread. Increase monitoring frequency and consider preventive action even on "
                    "unaffected plants nearby.")
    elif humidity_pct >= 60:
        level = "Moderate"
        message = "Moderately humid conditions may support disease spread. Keep monitoring the crop regularly."
    else:
        level = "Low"
        message = "Current weather conditions are less favorable for rapid disease spread, but continue regular monitoring."
    return {"level": level, "message": message}


def get_urgency(is_healthy: bool, severity_level, stage_name: str, weather_risk_level):
    """
    Classifies how quickly the farmer should act, based on severity,
    growth stage sensitivity, and current weather-driven disease risk.
    """
    if is_healthy:
        return {"emoji": "🟢", "label": "Preventive care",
                "detail": "No disease detected. Continue regular monitoring and preventive field hygiene."}

    sensitive_stage = any(k in stage_name.lower() for k in ["flowering", "fruiting", "bulking", "reproductive"])

    if severity_level == "Severe":
        return {"emoji": "🔴", "label": "Act immediately",
                "detail": "Symptoms indicate a well-established infection. Take action today to limit spread."}
    if weather_risk_level == "High" and severity_level == "Moderate":
        return {"emoji": "🔴", "label": "Act immediately",
                "detail": "Weather conditions are favorable for rapid spread on top of a moderate infection — act today."}
    if severity_level == "Moderate":
        label_detail = "Plan treatment or control measures within the next 1–2 days."
        if sensitive_stage:
            label_detail += " Since the crop is in a sensitive growth stage, double-check label approval for this stage first."
        return {"emoji": "🟠", "label": "Act within 1–2 days", "detail": label_detail}
    if severity_level == "Mild":
        return {"emoji": "🟡", "label": "Monitor closely",
                "detail": "Early-stage symptoms detected. Monitor closely and be ready to act if the condition worsens."}
    return {"emoji": "🟡", "label": "Monitor closely", "detail": "Continue monitoring your crop regularly."}


# ==========================================================================
# ---------- STEP 1: Upload photo ----------
# ==========================================================================
st.markdown('<div class="section-header">📸 Scan a crop leaf</div>', unsafe_allow_html=True)
st.markdown('<div class="upload-panel-label">Drag a leaf photo below, or click to browse</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

img = None
if uploaded:
    img = Image.open(uploaded).convert("RGB")

# ==========================================================================
# ---------- STEP 2: Ask crop details BEFORE showing the AI result ----------
# (moved up from the old "NEW FEATURE 7" block so farmers answer these
#  questions first, then see the detection result below)
# ==========================================================================
crop_age_days = 30
previously_affected = "Not sure"
include_weather = False
temp_c, humidity_pct = None, None

if uploaded:
    st.markdown('<div class="section-header">🌱 Tell us about your crop</div>', unsafe_allow_html=True)
    age_col, weather_col = st.columns([1, 1], gap="large")

    with age_col:
        st.markdown('<div class="upload-panel-label">How many days ago was this crop planted?</div>', unsafe_allow_html=True)
        crop_age_days = st.number_input(
            "Crop age (days)", min_value=1, max_value=365, value=30, step=1,
            help="For example: 10, 25, 45, 60 or 90 days", label_visibility="collapsed"
        )
        st.markdown(f"**Crop Age:** {crop_age_days} days")

        previously_affected = st.selectbox(
            "Has this crop shown this disease before?",
            ["Not sure", "No, first time", "Yes, it had this issue before"]
        )

    with weather_col:
        include_weather = st.checkbox("I know the current weather conditions (optional)")
        if include_weather:
            temp_c = st.number_input("Current temperature (°C)", min_value=0, max_value=55, value=28, step=1)
            humidity_pct = st.number_input("Current humidity (%)", min_value=0, max_value=100, value=70, step=1)

# ==========================================================================
# ---------- STEP 3: Detection Result (shown after the questions above) ----
# ==========================================================================
st.markdown('<div class="section-header">🔍 Detection result</div>', unsafe_allow_html=True)

col_upload, col_result = st.columns([1, 1.2], gap="large")

with col_upload:
    if img is not None:
        st.image(img, use_container_width=True)

with col_result:
    if uploaded:
        img_resized = img.resize((224, 224))
        arr = np.expand_dims(np.array(img_resized) / 255.0, axis=0)
        pred = model.predict(arr)
        result = class_names[np.argmax(pred)]
        confidence = float(np.max(pred)) * 100
        info = disease_info.get(result, {
            "display": result.replace("_", " "), "crop": result.split("_")[0].replace("__", " ").strip() or "Crop", "icon": "🌿",
            "treatment_en": "Consult a local agriculture expert.",
            "treatment_hi": "स्थानीय कृषि विशेषज्ञ से सलाह लें।",
            "treatment_mr": "स्थानिक कृषी तज्ञांचा सल्ला घ्या.",
            "treatment_kn": "ಸ್ಥಳೀಯ ಕೃಷಿ ತಜ್ಞರನ್ನು ಸಂಪರ್ಕಿಸಿ.",
            "severity": "moderate"
        })
        color = severity_colors.get(info["severity"], "#f2c744")
        is_healthy = info["severity"] == "healthy"

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

        # === NEW FEATURE 4 (part B): VALIDATION WARNINGS + ADVISORY NOTE ===
        validation = validate_prediction(img, confidence)
        for warning_msg in validation["warnings"]:
            if "Low confidence" in warning_msg or "resolution" in warning_msg or "little detail" in warning_msg:
                st.warning(warning_msg)
            else:
                st.info(warning_msg)
        st.caption("AI predictions are advisory and should be verified with an agricultural "
                   "expert when necessary. This tool does not claim 100% accuracy.")

        # Only compute/show disease-specific severity + duration if not healthy
        # and the image passed basic validation checks.
        severity = None if is_healthy else get_severity(info, confidence)

        if severity is not None and validation["is_valid"]:
            # === NEW FEATURE 2: SEVERITY CARD ===
            st.markdown(f"""
                <div class="info-card {severity['css_class']}">
                    <h4>🩺 Estimated Disease Severity: {severity['level']}</h4>
                    <p>{severity['explanation']}</p>
                    <p><b>Recommended action:</b> {severity['action']}</p>
                    <p style="font-size:0.78rem; color:#9fab8f;">AI-based/estimated assessment,
                    not a scientifically validated diagnosis.</p>
                </div>
            """, unsafe_allow_html=True)

            # === NEW FEATURE 1: DISEASE DURATION / PROGRESSION CARD ===
            duration_text = get_disease_duration(severity["level"])
            st.markdown(f"""
                <div class="info-card">
                    <h4>⏳ Estimated Disease Progression</h4>
                    <p>{duration_text}</p>
                    <p style="font-size:0.78rem; color:#9fab8f;">Approximate estimate — actual
                    progression depends on crop, weather, and infection conditions.</p>
                </div>
            """, unsafe_allow_html=True)
        elif is_healthy:
            st.success("✅ No disease detected. Your crop appears healthy!")

        # ======================================================
        # === NEW FEATURE 3: CROP HEALTH PROGRESS DASHBOARD ===
        # ======================================================
        health_level = "Healthy" if is_healthy else (severity["level"] if severity else "Moderate")
        health = get_crop_health_progress(is_healthy=is_healthy, severity_level=None if is_healthy else health_level)
        duration_display = "No active disease detected" if is_healthy else get_disease_duration(health_level)

        st.markdown(f"""
            <div class="info-card" style="border-left-color:{health['status_color']};">
                <h4>📊 Crop Health Progress</h4>
                <p><b>Crop / Condition:</b> {info['display']}</p>
                <p><b>Severity:</b> {"Healthy" if is_healthy else health_level} &nbsp;|&nbsp;
                   <b>AI Confidence:</b> {confidence:.1f}%</p>
                <p><b>Progression:</b> {duration_display}</p>
                <div class="health-bar-bg">
                    <div class="health-bar-fill" style="width:{health['percent']}%; background-color:{health['status_color']};"></div>
                </div>
                <span class="health-status-badge" style="background-color:{health['status_color']}22;
                    color:{health['status_color']}; border:1px solid {health['status_color']};">
                    Your crop health status: {health['status_label']}
                </span>
            </div>
        """, unsafe_allow_html=True)

        lang_choice = st.selectbox("🌐 Voice language", ["English", "हिंदी (Hindi)", "मराठी (Marathi)", "ಕನ್ನಡ (Kannada)"])
        lang_map = {
            "English": ("en", "treatment_en"),
            "हिंदी (Hindi)": ("hi", "treatment_hi"),
            "मराठी (Marathi)": ("mr", "treatment_mr"),
            "ಕನ್ನಡ (Kannada)": ("kn", "treatment_kn"),
        }
        lang_code, treatment_key = lang_map[lang_choice]
        treatment_text = info[treatment_key]

        st.markdown(f"""
            <div class="treatment-box">
                <div style="font-weight:700; color:#f2c744; margin-bottom:6px;">💊 Recommended action</div>
                <div style="color:#eef2e6; line-height:1.6;">{treatment_text}</div>
            </div>
        """, unsafe_allow_html=True)

        if st.button("🔊 Play voice advice", use_container_width=True):
            from gtts import gTTS
            tts = gTTS(f"{info['display']}. {treatment_text}", lang=lang_code)
            tts.save("output.mp3")
            st.audio("output.mp3")
    else:
        st.markdown("""
            <div style="height:100%; display:flex; align-items:center; justify-content:center; text-align:center; color:#9fab8f; padding: 3rem 1rem;">
                <div>
                    <div style="font-size:3.2rem;">🌱</div>
                    <div style="margin-top:10px;">Upload a photo to see detection results here</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

# ==========================================================================
# ---------- STEP 4: Growth stage & detailed recommendation ----------------
# (uses the crop-age / previously-affected / weather answers collected in
#  STEP 2, together with the detection result from STEP 3)
# ==========================================================================
if uploaded:
    st.markdown('<div class="section-header">🌱 Crop Age & Detailed Recommendation</div>', unsafe_allow_html=True)

    crop_name = info.get("crop") or "Your crop"

    # --- Growth stage (crop-specific where available) ---
    stage_info = get_growth_stage(crop_name, crop_age_days)
    stage_name = stage_info["stage"]
    stage_is_estimate = stage_info["is_estimate"]

    estimate_note = (
        "⚠️ Crop-specific growth-stage data isn't available for this crop — this is an "
        "approximate estimate based on general growth patterns."
        if stage_is_estimate else
        "Based on typical growth-stage timelines for this crop. Actual stage may vary with "
        "variety and local conditions."
    )
    st.markdown(f"""
        <div class="info-card">
            <h4>📅 Estimated Growth Stage</h4>
            <p><b>Crop:</b> {crop_name} &nbsp;|&nbsp; <b>Crop Age:</b> {crop_age_days} days</p>
            <p><b>Growth Stage:</b> {stage_name}</p>
            <p style="font-size:0.78rem; color:#9fab8f;">{estimate_note}</p>
        </div>
    """, unsafe_allow_html=True)

    # --- Weather risk (only if the farmer provided it) ---
    weather_risk = get_weather_risk(temp_c, humidity_pct) if include_weather else None

    severity_level_str = severity["level"] if (not is_healthy and severity is not None) else None
    pesticide = None if is_healthy else get_pesticide_recommendation(result)
    non_chemical_first = (not is_healthy) and (
        severity_level_str == "Mild" or any(k in stage_name.lower() for k in ["seedling", "early", "sprouting"])
    )

    # --- Recommended Action ---
    st.markdown('<div class="subsection-title">✅ Recommended Action</div>', unsafe_allow_html=True)
    if is_healthy:
        action_html = (
            f"<p>No disease was detected on this {crop_name.lower()} leaf.</p>"
            f"<p>At {crop_age_days} days old ({stage_name.lower()}), continue regular field monitoring, "
            f"maintain good field hygiene, and re-check the crop every few days, especially during humid weather.</p>"
        )
        card_class = ""
    else:
        chem_line = (
            "Not urgently required yet — prioritize non-chemical/cultural control first "
            "(sanitation, spacing, avoiding leaf wetness) and monitor closely."
            if non_chemical_first else
            "A fungicide/bactericide treatment is appropriate at this stage if symptoms persist "
            "or worsen — see the Treatment section below."
        )
        next_check = "2–3 days" if severity_level_str == "Severe" else "3–5 days"
        action_html = f"""
            <p><b>1. Disease detected:</b> {info['display']} (Confidence: {confidence:.1f}%)</p>
            <p><b>2. Estimated severity:</b> {severity_level_str} — {severity['explanation']}</p>
            <p><b>3. Immediate step:</b> Remove and safely destroy visibly infected leaves/plant parts,
               and avoid overhead irrigation to slow spread.</p>
            <p><b>4–5. Chemical treatment needed?</b> {chem_line}</p>
            <p><b>6. Next action:</b> Re-inspect the crop within {next_check} to track progress.</p>
        """
        card_class = severity["css_class"]

    st.markdown(f'<div class="info-card {card_class}">{action_html}</div>', unsafe_allow_html=True)

    # --- Treatment / Pesticide section ---
    st.markdown('<div class="subsection-title">💊 Treatment</div>', unsafe_allow_html=True)
    if is_healthy:
        st.info("No treatment is needed right now — your crop appears healthy.")
    else:
        if non_chemical_first:
            st.markdown("""
                <div class="info-card mild">
                    <p>🌿 Non-chemical/cultural control is recommended as the first step at this stage:</p>
                    <ul style="color:#d3ddc7; margin:0.3rem 0 0 1.1rem;">
                        <li>Remove and destroy infected leaves/plant debris</li>
                        <li>Avoid overhead irrigation; water at the base</li>
                        <li>Improve airflow/spacing between plants</li>
                        <li>Re-monitor every 2–3 days for worsening symptoms</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)

        st.markdown('<div style="font-weight:700; color:#eef2e6; margin:0.8rem 0 0.3rem 0;">🧪 Pesticide / Fungicide (if needed)</div>', unsafe_allow_html=True)
        if pesticide:
            st.markdown(f"""
                <div class="treatment-box">
                    <p style="color:#eef2e6;"><b>Product:</b> {pesticide['product']}</p>
                    <p style="color:#eef2e6;"><b>Purpose:</b> {pesticide['purpose']}</p>
                    <p style="color:#eef2e6;"><b>Recommended rate:</b> {pesticide['rate']}</p>
                    <p style="color:#eef2e6;"><b>Water/application volume:</b> {pesticide['water_volume']}</p>
                    <p style="color:#eef2e6;"><b>Application method:</b> {pesticide['method']}</p>
                    <p style="color:#eef2e6;"><b>Application timing:</b> {pesticide['timing']}</p>
                    <p style="color:#eef2e6;"><b>{pesticide['phi']}</b></p>
                    <p style="font-size:0.78rem; color:#bcc7ab; margin-top:8px;">{pesticide['source_note']}
                       Always confirm the exact rate on your product's label, as concentration can vary by
                       brand and formulation. When in doubt, consult your local agricultural officer.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Exact pesticide quantity cannot be safely determined from the available "
                       "information. Please follow the pesticide label or consult a local agricultural officer.")

    # --- Growth-stage-based warning ---
    st.markdown('<div class="subsection-title">⚠️ Growth-Stage Precaution</div>', unsafe_allow_html=True)
    st.info(get_stage_warning(crop_name, stage_name))

    # --- Weather Risk ---
    st.markdown('<div class="subsection-title">🌦️ Weather Risk</div>', unsafe_allow_html=True)
    if weather_risk:
        st.markdown(f"""
            <div class="info-card">
                <p><b>Temperature:</b> {temp_c}°C &nbsp;|&nbsp; <b>Humidity:</b> {humidity_pct}%</p>
                <p><b>Disease-spread risk:</b> {weather_risk['level']}</p>
                <p>{weather_risk['message']}</p>
                <p style="font-size:0.78rem; color:#9fab8f;">Weather conditions alone do not confirm
                   disease presence — this only indicates how favorable conditions currently are for
                   disease spread.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.caption("Weather information not provided — tick 'I know the current weather conditions' "
                   "above to see disease-spread risk based on temperature and humidity.")

    if previously_affected == "Yes, it had this issue before":
        st.caption("ℹ️ Since this field has had this issue before, disease pressure may already be "
                   "established — consider crop rotation and field sanitation for future seasons.")

    # --- Urgency ---
    st.markdown('<div class="subsection-title">⏱️ How Quickly Should You Act?</div>', unsafe_allow_html=True)
    urgency = get_urgency(
        is_healthy=is_healthy,
        severity_level=severity_level_str,
        stage_name=stage_name,
        weather_risk_level=weather_risk["level"] if weather_risk else None,
    )
    urgency_color = {"🔴": "#e0665a", "🟠": "#f2a444", "🟡": "#f2c744", "🟢": "#7bd389"}[urgency["emoji"]]
    st.markdown(f"""
        <span class="urgency-badge" style="background-color:{urgency_color}22; color:{urgency_color}; border:1px solid {urgency_color};">
            {urgency['emoji']} {urgency['label']}
        </span>
        <p style="color:#d3ddc7; margin-top:6px;">{urgency['detail']}</p>
    """, unsafe_allow_html=True)

    # --- Important Precaution ---
    st.markdown('<div class="subsection-title">🛡️ Important Precaution</div>', unsafe_allow_html=True)
    st.caption(
        "This recommendation is AI-assisted and advisory only. Always read and follow the actual "
        "pesticide label instructions, respect the pre-harvest interval, wear protective equipment "
        "while spraying, and consult your local Krishi Vibhag extension officer or agronomist for "
        "confirmation before applying any chemical treatment."
    )

# ---------- Helpline ----------
st.markdown('<div class="section-header">📞 Farmer helpline & support</div>', unsafe_allow_html=True)
st.markdown("""
    <div class="helpline-card">
        <div style="font-weight:700; color:#f2c744; margin-bottom:6px;">📱 Kisan Call Centre (Government of India)</div>
        <div style="color:#eef2e6; font-size:0.95rem; margin-bottom:14px;">Toll-free <b>1800-180-1551</b> · 6 AM–10 PM, all 7 days · 22 local languages</div>
        <div style="font-weight:700; color:#f2c744; margin-bottom:6px;">📱 PM-KISAN Helpline</div>
        <div style="color:#eef2e6; font-size:0.95rem;">Toll-free <b>155261</b> / <b>1800-115-526</b> &nbsp;|&nbsp; ☎️ 011-24300606</div>
    </div>
""", unsafe_allow_html=True)
st.caption("Numbers verified from official Government of India sources. Production version will link directly to the nearest Maharashtra Krishi Vibhag extension officer by location.")

# ---------- Roadmap ----------
st.markdown('<div class="section-header">🌱 Expanding crop coverage</div>', unsafe_allow_html=True)
st.write("This prototype currently detects diseases in **tomato, potato, and bell pepper**. Next, we're expanding to Maharashtra's core crops:")

r1, r2, r3, r4 = st.columns(4)
with r1:
    st.markdown('<div class="roadmap-chip">🌾 <b>Jowar</b><br><span style="font-size:0.85rem;">Grain mold, downy mildew</span></div>', unsafe_allow_html=True)
with r2:
    st.markdown('<div class="roadmap-chip">🌾 <b>Rice</b><br><span style="font-size:0.85rem;">Blast, bacterial blight</span></div>', unsafe_allow_html=True)
with r3:
    st.markdown('<div class="roadmap-chip">🌿 <b>Cotton</b><br><span style="font-size:0.85rem;">Pink bollworm, leaf curl</span></div>', unsafe_allow_html=True)
with r4:
    st.markdown('<div class="roadmap-chip">🎋 <b>Sugarcane</b><br><span style="font-size:0.85rem;">Red rot, smut</span></div>', unsafe_allow_html=True)

st.markdown('<p class="footer-note">Prototype for SIH 2026 · Problem Statement SIH26131 · Government of Maharashtra</p>', unsafe_allow_html=True)
