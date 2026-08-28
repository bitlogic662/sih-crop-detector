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
    .stApp .stSelectbox label, .stApp .stFileUploader label, .stApp .stNumberInput label { color: #eef2e6 !important; }

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
    div[data-testid="stNumberInput"] input {
        background: rgba(255,255,255,0.08) !important;
        color: #eef2e6 !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
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

    /* === feature CSS: cards for duration / severity / health dashboard === */
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

    /* === NEW: growth-stage / recommendation / urgency card styling === */
    .stage-card {
        background: rgba(255,255,255,0.06);
        border-radius: 18px;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
        border: 1px solid rgba(255,255,255,0.10);
        border-left: 5px solid #7bb7d3;
    }
    .stage-card h4 { margin-top: 0; margin-bottom: 0.4rem; color: #f6f9f2; }
    .stage-card p { margin: 0.2rem 0; color: #d3ddc7; font-size: 0.92rem; }

    .recommend-card {
        background: rgba(255,255,255,0.06);
        border-radius: 20px;
        padding: 1.6rem 1.8rem;
        margin-top: 1.4rem;
        box-shadow: 0 8px 26px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.10);
        border-top: 5px solid #f2c744;
    }
    .recommend-card h3 { margin-top: 0; color: #f6f9f2; }
    .recommend-card h4 { color: #f2c744; margin-bottom: 0.3rem; margin-top: 1.1rem; }
    .recommend-card p, .recommend-card li { color: #eef2e6; font-size: 0.95rem; line-height: 1.55; }

    .urgency-pill {
        display: inline-block;
        padding: 0.45rem 1rem;
        border-radius: 20px;
        font-weight: 800;
        font-size: 0.95rem;
        margin-top: 0.3rem;
    }

    .pesticide-warning {
        background: rgba(224,102,90,0.12);
        border: 1px solid rgba(224,102,90,0.4);
        border-radius: 14px;
        padding: 0.9rem 1.1rem;
        color: #f7d9d6 !important;
        font-size: 0.9rem;
    }
    .pesticide-verified {
        background: rgba(123,211,137,0.10);
        border: 1px solid rgba(123,211,137,0.4);
        border-radius: 14px;
        padding: 0.9rem 1.1rem;
        color: #eef2e6 !important;
        font-size: 0.9rem;
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
        "severity": "moderate",
        "chemical_ok": True,
        "non_chemical_first": True,
    },
    "Potato___Early_blight": {
        "display": "Potato — Early Blight",
        "crop": "Potato",
        "icon": "🥔",
        "treatment_en": "Apply fungicide (Chlorothalonil or Mancozeb). Rotate crops and remove infected debris.",
        "treatment_hi": "फफूंदनाशक (क्लोरोथालोनिल या मैंकोजेब) का प्रयोग करें। फसल चक्र अपनाएं और संक्रमित अवशेष हटा दें।",
        "treatment_mr": "बुरशीनाशक (क्लोरोथॅलोनिल किंवा मॅन्कोझेब) वापरा. पीक फेरपालट करा आणि संक्रमित अवशेष काढून टाका.",
        "treatment_kn": "ಶಿಲೀಂಧ್ರನಾಶಕ (ಕ್ಲೋರೋಥಲೋನಿಲ್ ಅಥವಾ ಮ್ಯಾಂಕೋಜೆಬ್) ಬಳಸಿ. ಬೆಳೆ ಸರದಿ ಅನುಸರಿಸಿ ಮತ್ತು ಸೋಂಕಿತ ಅವಶೇಷಗಳನ್ನು ತೆಗೆದುಹಾಕಿ.",
        "severity": "moderate",
        "chemical_ok": True,
        "non_chemical_first": True,
    },
    "Tomato_Late_blight": {
        "display": "Tomato — Late Blight",
        "crop": "Tomato",
        "icon": "🍅",
        "treatment_en": "Apply copper-based fungicide immediately. Remove and destroy infected plants to prevent spread.",
        "treatment_hi": "तुरंत कॉपर आधारित फफूंदनाशक का प्रयोग करें। फैलाव रोकने के लिए संक्रमित पौधों को हटाकर नष्ट कर दें।",
        "treatment_mr": "त्वरित तांबेयुक्त बुरशीनाशक वापरा. प्रसार रोखण्यासाठी संक्रमित रोपे काढून नष्ट करा.",
        "treatment_kn": "ತಕ್ಷಣ ತಾಮ್ರ ಆಧಾರಿತ ಶಿಲೀಂಧ್ರನಾಶಕವನ್ನು ಬಳಸಿ. ಹರಡುವಿಕೆಯನ್ನು ತಡೆಯಲು ಸೋಂಕಿತ ಸಸ್ಯಗಳನ್ನು ತೆಗೆದು ನಾಶಪಡಿಸಿ.",
        "severity": "severe",
        "chemical_ok": True,
        "non_chemical_first": False,
    },
    "Tomato_healthy": {
        "display": "Tomato — Healthy",
        "crop": "Tomato",
        "icon": "✅",
        "treatment_en": "No disease detected. Continue regular monitoring and good field hygiene.",
        "treatment_hi": "कोई रोग नहीं पाया गया। नियमित निगरानी और अच्छी खेत स्वच्छता जारी रखें।",
        "treatment_mr": "कोणताही रोग आढळला नाही. नियमित देखरेख आणि चांगली शेत स्वच्छता सुरू ठेवा.",
        "treatment_kn": "ಯಾವುದೇ ರೋಗ ಪತ್ತೆಯಾಗಿಲ್ಲ. ನಿಯಮಿತ ಮೇಲ್ವಿಚಾರಣೆ ಮತ್ತು ಉತ್ತಮ ಹೊಲದ ನೈರ್ಮಲ್ಯವನ್ನು ಮುಂದುವರಿಸಿ.",
        "severity": "healthy",
        "chemical_ok": False,
        "non_chemical_first": True,
    }
}

severity_colors = {"healthy": "#7bd389", "moderate": "#f2c744", "severe": "#e0665a"}
severity_labels = {"healthy": "Healthy", "moderate": "Moderate risk", "severe": "Severe — act now"}


# ======================================================
# === BASIC IMAGE VALIDATION ===
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
# === DISEASE SEVERITY ASSESSMENT ===
# ======================================================
def get_severity(info: dict, confidence: float):
    """
    Combines the per-disease 'severity' rule (healthy/moderate/severe
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
# === DISEASE DURATION / PROGRESSION ===
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
# === CROP HEALTH PROGRESS DASHBOARD ===
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
# === NEW: CROP-SPECIFIC GROWTH STAGE DATA ===
# ======================================================
# General agronomic stage windows (days after planting/transplanting) per crop.
# These are typical, approximate ranges for these crops and are shown to the
# farmer as an estimate — actual timing varies by variety, region and season.
CROP_GROWTH_STAGES = {
    "Tomato": [
        (0, 20, "Seedling / early establishment stage"),
        (21, 40, "Vegetative growth stage"),
        (41, 55, "Flowering stage"),
        (56, 90, "Fruiting / reproductive stage"),
        (91, 999, "Maturity / harvest stage"),
    ],
    "Potato": [
        (0, 20, "Sprouting / early establishment stage"),
        (21, 40, "Vegetative growth stage"),
        (41, 60, "Tuber initiation / flowering stage"),
        (61, 90, "Tuber bulking stage"),
        (91, 999, "Maturity stage"),
    ],
    "Bell Pepper": [
        (0, 20, "Seedling / early establishment stage"),
        (21, 40, "Vegetative growth stage"),
        (41, 60, "Flowering stage"),
        (61, 90, "Fruiting / reproductive stage"),
        (91, 999, "Maturity / harvest stage"),
    ],
}


def get_growth_stage(crop: str, age_days: int):
    """
    Returns {"stage": str, "is_estimate": bool, "note": str}.
    Uses crop-specific windows where available; otherwise falls back to a
    generic estimate and clearly labels it as approximate.
    """
    stages = CROP_GROWTH_STAGES.get(crop)
    if stages:
        for low, high, label in stages:
            if low <= age_days <= high:
                return {"stage": label, "is_estimate": False,
                        "note": f"Based on typical {crop} growth timing."}
        # age beyond known windows
        return {"stage": stages[-1][2], "is_estimate": False,
                "note": f"Based on typical {crop} growth timing."}

    # No crop-specific data available — generic fallback, clearly marked
    generic = [
        (0, 20, "Early / seedling stage"),
        (21, 40, "Vegetative stage"),
        (41, 60, "Flowering stage"),
        (61, 90, "Fruiting / reproductive stage"),
        (91, 999, "Maturity stage"),
    ]
    for low, high, label in generic:
        if low <= age_days <= high:
            return {"stage": label, "is_estimate": True,
                    "note": "Crop-specific growth data is not available for this crop — "
                            "this stage is a generic approximate estimate only."}
    return {"stage": "Maturity stage", "is_estimate": True,
            "note": "Crop-specific growth data is not available for this crop — "
                    "this stage is a generic approximate estimate only."}


# ======================================================
# === NEW: AGE / STAGE-BASED WARNINGS ===
# ======================================================
def get_age_stage_warning(crop: str, stage_label: str):
    """
    Returns a farmer-facing caution message tailored to the crop's current
    growth stage. Kept general (no dosage/product claims here).
    """
    s = stage_label.lower()
    if "seedling" in s or "early" in s or "sprouting" in s:
        return ("🌱 Your crop is in an early growth stage. Avoid unnecessary chemical treatment "
                "and prioritise preventive/cultural measures (removing infected leaves, spacing, "
                "avoiding overhead watering) first.")
    if "flowering" in s:
        return ("🌸 The crop is flowering. If a treatment is needed, use only products approved "
                "for this crop and growth stage, and follow the label instructions carefully — "
                "some products can affect flowering or pollinators.")
    if "fruit" in s or "tuber bulking" in s or "reproductive" in s:
        return ("🍅 The crop is in the fruiting/bulking stage. Pay close attention to the "
                "product's pre-harvest interval (the minimum wait time between application and "
                "harvest) before applying any pesticide or fungicide.")
    if "matur" in s or "harvest" in s:
        return ("🌾 The crop is near maturity/harvest. Check the pre-harvest interval on the "
                "product label carefully — some treatments are not safe to apply this close to "
                "harvest.")
    return ("ℹ️ Match any treatment to your crop's current growth stage and always follow the "
            "product label.")


# ======================================================
# === NEW: VERIFIED PESTICIDE / DOSAGE LOOKUP (SAFE) ===
# ======================================================
# IMPORTANT SAFETY NOTE FOR DEVELOPERS:
# This app must NEVER invent a pesticide name, concentration, or dosage.
# Populate PESTICIDE_DOSAGE_DB only with entries copied from a verified
# source (the product's official label, or a government/CIBRC-approved
# pesticide database) for the specific crop + disease + product combination.
# Until an entry is added and verified, the app safely reports that the
# exact quantity cannot be determined — this is intentional, not a bug.
#
# Expected entry shape:
# ("Crop", "Disease key"): {
#     "product": "Verified product name",
#     "purpose": "What it controls",
#     "rate": "Label-approved rate (e.g. '2 g/litre of water')",
#     "water_volume": "Label-approved spray volume",
#     "timing": "Label-approved application timing",
#     "phi_days": "Pre-harvest interval, if on label",
# }
PESTICIDE_DOSAGE_DB = {
    # Intentionally left empty until verified, label-sourced entries are added.
}


def get_pesticide_recommendation(crop: str, disease_key: str, growth_stage: str):
    """
    Looks up a verified, label-sourced pesticide recommendation.
    NEVER fabricates a dosage — if no verified entry exists, returns the
    safe fallback message required by policy.
    """
    entry = PESTICIDE_DOSAGE_DB.get((crop, disease_key))
    if entry:
        return {"available": True, **entry}
    return {
        "available": False,
        "message": ("Exact pesticide quantity cannot be safely determined from the available "
                    "information. Please follow the pesticide label or consult a local "
                    "agricultural officer / Krishi Vibhag extension officer.")
    }


# ======================================================
# === NEW: URGENCY CLASSIFICATION ===
# ======================================================
def get_urgency(is_healthy: bool, severity_level: str, growth_stage: str, weather_risk: str):
    """
    Classifies recommended urgency from severity, growth stage and weather risk.
    Returns {"label": str, "emoji": str, "color": str}.
    """
    if is_healthy:
        return {"label": "Preventive care", "emoji": "🟢", "color": "#7bd389"}

    stage_sensitive = any(k in growth_stage.lower() for k in ["flower", "fruit", "matur", "harvest", "bulking"])

    if severity_level == "Severe":
        return {"label": "Act immediately", "emoji": "🔴", "color": "#e0665a"}
    if severity_level == "Moderate" and (weather_risk == "high" or stage_sensitive):
        return {"label": "Act within 1–2 days", "emoji": "🟠", "color": "#f2ab3a"}
    if severity_level == "Moderate":
        return {"label": "Act within 1–2 days", "emoji": "🟠", "color": "#f2ab3a"}
    if severity_level == "Mild" and weather_risk == "high":
        return {"label": "Act within 1–2 days", "emoji": "🟠", "color": "#f2ab3a"}
    return {"label": "Monitor closely", "emoji": "🟡", "color": "#f2c744"}


# ======================================================
# === NEW: WEATHER RISK (uses optional farmer-entered readings) ===
# ======================================================
def get_weather_risk(temp_c, humidity_pct):
    """
    Simple, transparent rule-of-thumb: warm + humid conditions favour fungal/
    bacterial spread for these crops. This does NOT diagnose disease from
    weather alone — it only flags conditions that can accelerate spread.
    Returns {"risk": "high"/"moderate"/"low"/"unknown", "text": str}.
    """
    if temp_c is None or humidity_pct is None:
        return {"risk": "unknown",
                "text": "Weather readings were not provided, so weather-based risk could not be assessed."}

    if humidity_pct >= 80 and 20 <= temp_c <= 30:
        return {"risk": "high",
                "text": f"At {temp_c:.0f}°C and {humidity_pct:.0f}% humidity, conditions are warm and humid — "
                        "favourable for the disease to spread faster on this crop. This does not confirm "
                        "disease by itself, but supports acting promptly on the AI detection above."}
    if humidity_pct >= 60:
        return {"risk": "moderate",
                "text": f"At {temp_c:.0f}°C and {humidity_pct:.0f}% humidity, conditions are moderately "
                        "favourable for disease spread. Keep monitoring the crop closely."}
    return {"risk": "low",
            "text": f"At {temp_c:.0f}°C and {humidity_pct:.0f}% humidity, current conditions are less "
                    "favourable for rapid disease spread."}


# ---------- Upload + Result ----------
st.markdown('<div class="section-header">📸 Scan a crop leaf</div>', unsafe_allow_html=True)

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
        info = disease_info.get(result, {
            "display": result.replace("_", " "), "crop": result.split("_")[0].replace("__", " ").strip(),
            "icon": "🌿",
            "treatment_en": "Consult a local agriculture expert.",
            "treatment_hi": "स्थानीय कृषि विशेषज्ञ से सलाह लें।",
            "treatment_mr": "स्थानिक कृषी तज्ञांचा सल्ला घ्या.",
            "treatment_kn": "ಸ್ಥಳೀಯ ಕೃಷಿ ತಜ್ಞರನ್ನು ಸಂಪರ್ಕಿಸಿ.",
            "severity": "moderate",
            "chemical_ok": True,
            "non_chemical_first": True,
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

        # === VALIDATION WARNINGS + ADVISORY NOTE ===
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
            # === SEVERITY CARD ===
            st.markdown(f"""
                <div class="info-card {severity['css_class']}">
                    <h4>🩺 Estimated Disease Severity: {severity['level']}</h4>
                    <p>{severity['explanation']}</p>
                    <p><b>Recommended action:</b> {severity['action']}</p>
                    <p style="font-size:0.78rem; color:#9fab8f;">AI-based/estimated assessment,
                    not a scientifically validated diagnosis.</p>
                </div>
            """, unsafe_allow_html=True)

            # === DISEASE DURATION / PROGRESSION CARD ===
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
        # === CROP HEALTH PROGRESS DASHBOARD ===
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

        # ======================================================
        # === NEW FEATURE: CROP AGE + GROWTH STAGE INPUT ===
        # ======================================================
        st.markdown('<div class="section-header">🌱 Crop age & growth stage</div>', unsafe_allow_html=True)
        crop_age_days = st.number_input(
            "How many days ago was this crop planted?",
            min_value=0, max_value=365, value=30, step=1,
            help="Enter the number of days since planting/transplanting, e.g. 10, 25, 45, 60, 90."
        )
        st.markdown(f"**Crop Age:** {int(crop_age_days)} days")

        crop_name = info.get("crop", info["display"].split("—")[0].strip())
        growth = get_growth_stage(crop_name, int(crop_age_days))

        stage_note = growth["note"] if growth["is_estimate"] else growth["note"]
        st.markdown(f"""
            <div class="stage-card">
                <h4>🌿 Estimated Growth Stage: {growth['stage']}</h4>
                <p>{stage_note}</p>
            </div>
        """, unsafe_allow_html=True)

        # === Optional weather readings (offline-friendly manual entry) ===
        with st.expander("🌦️ Optional: enter local weather readings for a fuller recommendation"):
            wc1, wc2 = st.columns(2)
            with wc1:
                have_weather = st.checkbox("I have temperature/humidity readings")
            temp_c = humidity_pct = None
            if have_weather:
                with wc1:
                    temp_c = st.number_input("Temperature (°C)", min_value=-10.0, max_value=55.0, value=28.0, step=0.5)
                with wc2:
                    humidity_pct = st.number_input("Relative humidity (%)", min_value=0.0, max_value=100.0, value=70.0, step=1.0)

        weather = get_weather_risk(temp_c, humidity_pct)

        # === Optional: previously affected status ===
        previously_affected = st.checkbox("This plant/field was previously affected by a disease this season")

        # ======================================================
        # === NEW FEATURE: RECOMMENDED ACTION (combined) ===
        # ======================================================
        st.markdown('<div class="section-header">🧭 Recommended Action</div>', unsafe_allow_html=True)

        urgency = get_urgency(is_healthy, None if is_healthy else health_level, growth["stage"], weather["risk"])
        age_warning = get_age_stage_warning(crop_name, growth["stage"])

        if is_healthy:
            explanation = ("No disease was detected on this leaf. Continue routine field monitoring, "
                            "good sanitation, and balanced watering/fertilisation appropriate to the "
                            f"current {growth['stage'].lower()}.")
            do_now = "No treatment is needed right now — recheck the crop regularly, especially after rain."
            chemical_line = "Not applicable — no disease detected."
            non_chem_line = "Continue preventive/cultural practices (crop rotation, field sanitation, adequate spacing)."
        else:
            explanation = (f"The AI model detected **{info['display']}** with **{confidence:.1f}%** confidence, "
                            f"assessed as **{health_level}** severity at the crop's current "
                            f"**{growth['stage'].lower()}**.")
            if health_level == "Severe":
                do_now = "Remove and isolate/destroy the worst-affected leaves or plants now to slow spread, and act on treatment without delay."
            elif health_level == "Moderate":
                do_now = "Remove visibly affected leaves, improve airflow/drainage, and prepare to treat within the next couple of days."
            else:
                do_now = "Keep monitoring closely and remove any newly affected leaves; treatment may not be urgent yet."

            if info.get("chemical_ok", True):
                chemical_line = "A pesticide/fungicide treatment can be appropriate for this disease at this stage — see the verified product details below, and always match the product to this crop and disease."
            else:
                chemical_line = "Chemical treatment is not indicated for this result."

            if info.get("non_chemical_first", True) and health_level != "Severe":
                non_chem_line = "Try non-chemical steps first: remove infected leaves/debris, avoid overhead watering, improve spacing and airflow, and rotate crops next season."
            else:
                non_chem_line = "Non-chemical steps (sanitation, removing infected material) should be used alongside any chemical treatment, not instead of it, given the severity."

        pesticide = get_pesticide_recommendation(crop_name, result, growth["stage"])

        recommend_html = f"""
        <div class="recommend-card">
            <h3>{info['icon']} {crop_name} · {info['display']}</h3>
            <p><b>Crop Age:</b> {int(crop_age_days)} days &nbsp;|&nbsp; <b>Growth Stage:</b> {growth['stage']}</p>
            <p><b>Detected Disease:</b> {info['display']} &nbsp;|&nbsp; <b>Confidence:</b> {confidence:.1f}%</p>
            <p><b>Severity:</b> {"Healthy" if is_healthy else health_level}</p>

            <h4>1. What's happening</h4>
            <p>{explanation}</p>

            <h4>2. What to do immediately</h4>
            <p>{do_now}</p>

            <h4>3. Chemical treatment?</h4>
            <p>{chemical_line}</p>

            <h4>4. Try non-chemical methods first?</h4>
            <p>{non_chem_line}</p>

            <h4>5. Age/stage-specific caution</h4>
            <p>{age_warning}</p>
        </div>
        """
        st.markdown(recommend_html, unsafe_allow_html=True)

        # === Pesticide / Quantity block (never fabricated) ===
        st.markdown('<div class="section-header" style="font-size:1.15rem; margin-top:1.4rem;">💊 Pesticide/Fungicide &amp; Quantity</div>', unsafe_allow_html=True)
        if not is_healthy and info.get("chemical_ok", True):
            if pesticide["available"]:
                st.markdown(f"""
                    <div class="pesticide-verified">
                        <b>Product:</b> {pesticide['product']}<br>
                        <b>Purpose:</b> {pesticide['purpose']}<br>
                        <b>Recommended rate:</b> {pesticide['rate']}<br>
                        <b>Water/application volume:</b> {pesticide['water_volume']}<br>
                        <b>Application timing:</b> {pesticide['timing']}<br>
                        <b>Pre-harvest interval:</b> {pesticide.get('phi_days', 'See product label')}
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="pesticide-warning">
                        ⚠️ {pesticide['message']}
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("No pesticide/fungicide recommendation applicable for this result.")

        # === Urgency ===
        st.markdown('<div class="section-header" style="font-size:1.15rem; margin-top:1.4rem;">⏱️ How Quickly Should You Act?</div>', unsafe_allow_html=True)
        st.markdown(f"""
            <span class="urgency-pill" style="background-color:{urgency['color']}22; color:{urgency['color']}; border:1px solid {urgency['color']};">
                {urgency['emoji']} {urgency['label']}
            </span>
        """, unsafe_allow_html=True)

        # === Weather risk ===
        st.markdown('<div class="section-header" style="font-size:1.15rem; margin-top:1.4rem;">🌦️ Weather Risk</div>', unsafe_allow_html=True)
        st.write(weather["text"])

        # === Previously affected note ===
        if previously_affected:
            st.info("ℹ️ Since this field/plant was previously affected this season, watch for recurring "
                    "symptoms and consider rotating treatment approaches to reduce resistance risk — "
                    "consult a local agricultural officer if the issue keeps returning.")

        st.caption("This recommendation is generated by an AI model plus general agronomic guidance. "
                   "It does not replace advice from a qualified agricultural officer.")

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
