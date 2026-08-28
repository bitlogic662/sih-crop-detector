import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, UnidentifiedImageError
import numpy as np
import json
from collections import Counter

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
        margin-bottom: 1.2rem;
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
    .result-name { font-size: 1.5rem; font-weight: 800; margin-top: 4px; color: #f6f9f2; }
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
    div[data-testid="stFormSubmitButton"] button {
        background: #f2c744 !important;
        color: #23331b !important;
        border: none !important;
        border-radius: 30px !important;
        font-weight: 800 !important;
        padding: 0.7rem 1.6rem !important;
        font-size: 1.02rem !important;
    }
    div[data-testid="stFormSubmitButton"] button:hover { background: #f6d768 !important; }

    /* === cards for duration / severity / health dashboard === */
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

    /* === crop age / growth stage / recommendation section === */
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
    div[data-testid="stNumberInput"] label, div[data-testid="stCheckbox"] label, div[data-testid="stRadio"] label, div[data-testid="stTextInput"] label {
        color: #eef2e6 !important;
    }
    div[data-testid="stNumberInput"] input, div[data-testid="stTextInput"] input {
        background: rgba(255,255,255,0.08) !important;
        color: #eef2e6 !important;
        border-radius: 10px !important;
    }

    /* Photo thumbnail grid for multi-upload */
    .photo-count-badge {
        display: inline-block;
        background: rgba(242,199,68,0.15);
        color: #f2c744;
        border: 1px solid rgba(242,199,68,0.4);
        border-radius: 30px;
        padding: 6px 16px;
        font-weight: 700;
        font-size: 0.9rem;
        margin-bottom: 12px;
    }
    .photo-thumb-label {
        text-align: center;
        font-size: 0.78rem;
        color: #d3ddc7;
        margin-top: 4px;
    }
    .mini-metric-card {
        background: rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 0.9rem 0.7rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.10);
    }
    .mini-metric-num { font-size: 1.5rem; font-weight: 800; color: #f2c744; }
    .mini-metric-label { font-size: 0.75rem; color: #d3ddc7; margin-top: 2px; }
    </style>
""", unsafe_allow_html=True)

# ---------- Load model ----------
@st.cache_resource
def load_my_model():
    model = load_model("crop_model.h5")
    with open("class_names.json") as f:
        class_names = json.load(f)
    return model, class_names

try:
    model, class_names = load_my_model()
    MODEL_LOAD_ERROR = None
except Exception as e:
    model, class_names = None, []
    MODEL_LOAD_ERROR = str(e)

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
    st.markdown('<div class="step-card"><div class="step-num">1</div><b>📷 Snap photos</b><br><span style="opacity:0.85; font-size:0.85rem;">Take clear photos of the affected leaves</span></div>', unsafe_allow_html=True)
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


def get_disease_info(result_key):
    """Fallback-safe lookup so an unrecognized class name never crashes the app."""
    return disease_info.get(result_key, {
        "display": result_key.replace("_", " ") if result_key else "Unknown condition",
        "crop": (result_key.split("_")[0].replace("__", " ").strip() if result_key else "Crop") or "Crop",
        "icon": "🌿",
        "treatment_en": "Consult a local agriculture expert.",
        "treatment_hi": "स्थानीय कृषि विशेषज्ञ से सलाह लें।",
        "treatment_mr": "स्थानिक कृषी तज्ञांचा सल्ला घ्या.",
        "treatment_kn": "ಸ್ಥಳೀಯ ಕೃಷಿ ತಜ್ಞರನ್ನು ಸಂಪರ್ಕಿಸಿ.",
        "severity": "moderate"
    })


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
def _severity_details(level):
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
    return details[level]


def get_severity(info: dict, confidence: float):
    """
    Combines the per-disease 'severity' rule (healthy/moderate/severe in
    disease_info) with model confidence to produce a Mild/Moderate/Severe
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

    d = _severity_details(level)
    return {"level": level, "explanation": d["explanation"], "action": d["action"], "css_class": d["css_class"]}


def get_severity_enhanced(info: dict, confidence: float, farmer_info: dict = None):
    """
    Wraps get_severity() and additionally nudges the estimated level using
    farmer-reported context (how much of the crop is affected, and how long
    symptoms have been noticed). This stays an AI-based ESTIMATE — it is not
    a scientifically validated severity measurement, and farmer answers alone
    never override what the model actually sees in the image.
    """
    severity = get_severity(info, confidence)
    if severity is None or not farmer_info:
        return severity

    levels = ["Mild", "Moderate", "Severe"]
    idx = levels.index(severity["level"])
    bump = 0

    affected_area = farmer_info.get("affected_area")
    if affected_area in ["50–75%", "More than 75%"]:
        bump += 1
    elif affected_area in ["Only one/few leaves", "Less than 25%"]:
        bump -= 1

    symptom_duration = farmer_info.get("symptom_duration")
    if symptom_duration in ["1–2 weeks", "More than 2 weeks"]:
        bump += 1
    elif symptom_duration == "Less than 1 day":
        bump -= 1

    if bump >= 2:
        idx = min(2, idx + 1)
    elif bump <= -2:
        idx = max(0, idx - 1)

    new_level = levels[idx]
    d = _severity_details(new_level)
    return {"level": new_level, "explanation": d["explanation"], "action": d["action"], "css_class": d["css_class"]}


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
    Converts a detection result into a simple visual health percentage
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
# === CROP-SPECIFIC GROWTH STAGE LOOKUP (fallback when farmer is "Not sure") ===
# ======================================================
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
    """Returns an estimated growth stage for a given crop and age in days."""
    stages = CROP_GROWTH_STAGES.get(crop)
    if stages:
        for lo, hi, name in stages:
            if lo <= age_days <= hi:
                return {"stage": name, "is_estimate": False}
        return {"stage": stages[-1][2], "is_estimate": False}

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
    """Generates a growth-stage-specific caution message."""
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
# === VERIFIED PESTICIDE/FUNGICIDE REFERENCE DATABASE ===
# ======================================================
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


def get_weather_risk_from_condition(condition: str):
    """
    Maps the farmer's plain-language field-condition answer to a
    disease-favorability read. Heuristic context only — never proof
    of disease presence.
    """
    mapping = {
        "High rainfall": ("High", "Recent high rainfall can keep leaves wet for long periods, which "
                                   "favors fungal and bacterial spread. Increase monitoring frequency."),
        "High humidity": ("High", "High humidity is favorable for fungal/bacterial disease spread. "
                                   "Increase monitoring frequency and consider preventive action on nearby plants."),
        "Very hot": ("Moderate", "Hot, dry conditions can stress plants and sometimes slow fungal spread, "
                                  "but can favor certain pests. Keep monitoring regularly."),
        "Very dry": ("Low", "Dry conditions are generally less favorable for fungal/bacterial spread, "
                             "but continue regular monitoring."),
        "Normal": ("Low", "Normal field conditions reported — continue regular monitoring."),
        "Not sure": (None, "Field condition not specified — disease-spread risk from weather could not be estimated."),
    }
    return mapping.get(condition, (None, "Field condition not specified — disease-spread risk from weather could not be estimated."))


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


# ======================================================
# === CORE MULTI-PHOTO PIPELINE FUNCTIONS ===
# ======================================================
def predict_single_image(image: Image.Image, model, class_names):
    """
    Runs the existing TensorFlow/Keras model on a single PIL image, using the
    same preprocessing as before (resize to 224x224, scale to 0-1).
    Returns (result_key, confidence, error_message). error_message is None on success.
    """
    try:
        img_resized = image.convert("RGB").resize((224, 224))
        arr = np.expand_dims(np.array(img_resized) / 255.0, axis=0)
        pred = model.predict(arr, verbose=0)
        idx = int(np.argmax(pred))
        if idx >= len(class_names):
            return None, 0.0, "Model returned an unrecognized class index."
        result_key = class_names[idx]
        confidence = float(np.max(pred)) * 100
        return result_key, confidence, None
    except Exception as e:
        return None, 0.0, f"Prediction failed: {e}"


def analyze_multiple_images(images, filenames, model, class_names, farmer_info):
    """
    Loops over every uploaded image, runs prediction + validation + severity +
    progression + health scoring, and returns a list of per-photo result dicts.
    The model is loaded once (via @st.cache_resource) and reused for every photo.
    """
    results = []
    for i, (image, fname) in enumerate(zip(images, filenames)):
        entry = {"index": i + 1, "filename": fname, "image": image, "prediction_ok": False}

        if model is None:
            entry["error"] = "Model is not loaded, so this photo could not be analyzed."
            results.append(entry)
            continue

        result_key, confidence, error = predict_single_image(image, model, class_names)
        if error:
            entry["error"] = error
            results.append(entry)
            continue

        info = get_disease_info(result_key)
        is_healthy = info.get("severity") == "healthy"
        validation = validate_prediction(image, confidence)
        severity = None if is_healthy else get_severity_enhanced(info, confidence, farmer_info)
        duration_text = None if is_healthy else get_disease_duration(severity["level"] if severity else None)
        health = get_crop_health_progress(is_healthy=is_healthy, severity_level=None if is_healthy else (severity["level"] if severity else None))

        entry.update({
            "prediction_ok": True,
            "result_key": result_key,
            "confidence": confidence,
            "info": info,
            "is_healthy": is_healthy,
            "validation": validation,
            "severity": severity,
            "duration_text": duration_text,
            "health": health,
        })
        results.append(entry)
    return results


def calculate_overall_assessment(results):
    """
    Combines predictions from ALL uploaded images (not just the
    highest-confidence one) into one overall assessment.
    """
    valid = [r for r in results if r.get("prediction_ok")]
    total = len(results)
    if not valid:
        return {"total": total, "valid_total": 0}

    healthy_count = sum(1 for r in valid if r["is_healthy"])
    affected_count = len(valid) - healthy_count
    avg_confidence = sum(r["confidence"] for r in valid) / len(valid)

    disease_counter = Counter(r["info"]["display"] for r in valid)
    most_common_disease, most_common_count = disease_counter.most_common(1)[0]

    severity_rank = {"Mild": 1, "Moderate": 2, "Severe": 3}
    affected_results = [r for r in valid if not r["is_healthy"] and r.get("severity")]
    if affected_results:
        sev_counter = Counter(r["severity"]["level"] for r in affected_results)
        # majority level; ties broken toward the more severe level (safer default)
        overall_level = max(sev_counter.items(), key=lambda kv: (kv[1], severity_rank[kv[0]]))[0]
        # find the disease_info entry matching the most common disease for pesticide/treatment lookup
        overall_result_key = Counter(r["result_key"] for r in affected_results).most_common(1)[0][0]
    else:
        overall_level = None
        overall_result_key = None

    overall_health = get_crop_health_progress(
        is_healthy=(affected_count == 0),
        severity_level=overall_level
    )

    unique_crops = sorted(set(r["info"]["crop"] for r in valid))

    return {
        "total": total,
        "valid_total": len(valid),
        "healthy_count": healthy_count,
        "affected_count": affected_count,
        "avg_confidence": avg_confidence,
        "most_common_disease": most_common_disease,
        "most_common_count": most_common_count,
        "overall_level": overall_level,
        "overall_result_key": overall_result_key,
        "overall_health": overall_health,
        "mixed_crops": len(unique_crops) > 1,
        "unique_crops": unique_crops,
        "affected_pct": round((affected_count / len(valid)) * 100, 1),
    }


def display_image_result(entry, lang_choice, lang_map):
    """Renders a single result card for one uploaded photo, matching the existing theme."""
    thumb_col, info_col = st.columns([1, 2], gap="medium")
    with thumb_col:
        st.image(entry["image"], use_container_width=True)
        st.markdown(f'<div class="photo-thumb-label">📷 Photo {entry["index"]} — {entry["filename"]}</div>', unsafe_allow_html=True)

    with info_col:
        if not entry.get("prediction_ok"):
            st.error(f"📷 Photo {entry['index']}: {entry.get('error', 'Could not be analyzed.')}")
            return

        info = entry["info"]
        confidence = entry["confidence"]
        is_healthy = entry["is_healthy"]
        severity = entry["severity"]
        color = severity_colors.get(info["severity"], "#f2c744")
        severity_label = "Healthy" if is_healthy else (severity["level"] if severity else "Unknown")

        st.markdown(f"""
            <div class="result-card" style="border-top-color:{color};">
                <div class="result-label">📷 Photo {entry['index']} — Detection result</div>
                <div class="result-name">{info['icon']} {info['display']}</div>
                <div style="margin-top:10px;">
                    <span class="status-dot" style="background-color:{color};"></span>
                    <span style="font-weight:600; color:{color};">{"Healthy" if is_healthy else severity_labels.get(info['severity'], '')}</span>
                </div>
                <div style="margin-top:14px; font-size:0.85rem; color:#d3ddc7;">Confidence: <b>{confidence:.1f}%</b></div>
                <div class="confidence-bar-bg">
                    <div class="confidence-bar-fill" style="width:{confidence}%; background-color:{color};"></div>
                </div>
                <p style="margin-top:12px; font-size:0.85rem;"><b>Severity:</b> {severity_label} &nbsp;|&nbsp;
                   <b>Progression:</b> {"Not applicable" if is_healthy else entry.get("duration_text", "N/A")} &nbsp;|&nbsp;
                   <b>Crop health:</b> {entry["health"]["status_label"]}</p>
            </div>
        """, unsafe_allow_html=True)

        for warning_msg in entry["validation"]["warnings"]:
            msg = warning_msg
            if "Low confidence" in warning_msg:
                msg = f"⚠️ Low confidence prediction for Photo {entry['index']}. Please upload a clearer close-up image of the affected leaf."
            if "resolution" in warning_msg or "little detail" in warning_msg:
                st.warning(f"Photo {entry['index']}: {msg}")
            else:
                st.info(f"Photo {entry['index']}: {msg}")

        lang_code, treatment_key = lang_map[lang_choice]
        st.markdown(f"""
            <div class="treatment-box">
                <div style="font-weight:700; color:#f2c744; margin-bottom:6px;">💊 Recommended action</div>
                <div style="color:#eef2e6; line-height:1.6;">{info[treatment_key]}</div>
            </div>
        """, unsafe_allow_html=True)


def display_overall_dashboard(overall):
    """Renders the combined 'Overall Crop Health Assessment' section."""
    if overall.get("valid_total", 0) == 0:
        st.error("None of the uploaded photos could be analyzed. Please try uploading clearer images.")
        return

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f'<div class="mini-metric-card"><div class="mini-metric-num">{overall["total"]}</div><div class="mini-metric-label">📷 Photos analyzed</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="mini-metric-card"><div class="mini-metric-num">{overall["affected_count"]}</div><div class="mini-metric-label">🦠 Affected photos</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="mini-metric-card"><div class="mini-metric-num">{overall["healthy_count"]}</div><div class="mini-metric-label">✅ Healthy photos</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="mini-metric-card"><div class="mini-metric-num">{overall["avg_confidence"]:.1f}%</div><div class="mini-metric-label">🎯 Avg. confidence</div></div>', unsafe_allow_html=True)
    with m5:
        st.markdown(f'<div class="mini-metric-card"><div class="mini-metric-num">{overall["overall_health"]["status_label"]}</div><div class="mini-metric-label">⚠️ Overall risk</div></div>', unsafe_allow_html=True)

    st.write("")
    health = overall["overall_health"]
    st.markdown(f"""
        <div class="info-card" style="border-left-color:{health['status_color']};">
            <h4>🌾 Overall Crop Health</h4>
            <p><b>Photos analyzed:</b> {overall['total']}</p>
            <p><b>Most detected condition:</b> {overall['most_common_disease']} ({overall['most_common_count']}/{overall['valid_total']} photos)</p>
            <p><b>Average confidence:</b> {overall['avg_confidence']:.1f}%</p>
            <p><b>Affected photos:</b> {overall['affected_count']}/{overall['valid_total']} ({overall['affected_pct']}%) &nbsp;|&nbsp;
               <b>Healthy photos:</b> {overall['healthy_count']}/{overall['valid_total']}</p>
            <p><b>Overall severity:</b> {overall['overall_level'] or 'Not applicable (no disease detected)'}</p>
            <p><b>Overall crop health status:</b> {health['status_label']}</p>
            <div class="health-bar-bg">
                <div class="health-bar-fill" style="width:{health['percent']}%; background-color:{health['status_color']};"></div>
            </div>
            <p style="font-size:0.78rem; color:#9fab8f; margin-top:8px;">This combines results from every uploaded
               photo, not just the single highest-confidence image. It is an AI-based estimate and should be
               verified with a local agricultural expert when necessary.</p>
        </div>
    """, unsafe_allow_html=True)

    if overall.get("mixed_crops"):
        st.warning("ℹ️ The uploaded photos appear to show more than one crop type: "
                   + ", ".join(overall["unique_crops"]) +
                   ". For the most reliable overall assessment, try analyzing photos of one crop at a time.")


# ======================================================
# === SESSION STATE INITIALIZATION ===
# ======================================================
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "overall_assessment" not in st.session_state:
    st.session_state.overall_assessment = None
if "farmer_info" not in st.session_state:
    st.session_state.farmer_info = None

if MODEL_LOAD_ERROR:
    st.error(f"⚠️ The disease-detection model could not be loaded ({MODEL_LOAD_ERROR}). "
             "Photo upload and farmer-information sections will still work, but analysis is unavailable "
             "until the model files are available.")

# ---------- Upload ----------
st.markdown('<div class="section-header">📸 Scan crop leaves</div>', unsafe_allow_html=True)
st.markdown('<div class="upload-panel-label">Drag one or more leaf photos below, or click to browse</div>', unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)

valid_images, valid_filenames = [], []
if uploaded_files:
    st.markdown(f'<div class="photo-count-badge">📷 Photos selected: {len(uploaded_files)}</div>', unsafe_allow_html=True)

    thumb_cols = st.columns(min(len(uploaded_files), 5) or 1)
    for i, f in enumerate(uploaded_files):
        try:
            img = Image.open(f).convert("RGB")
        except (UnidentifiedImageError, OSError):
            st.error(f"❌ Photo {i + 1} ('{f.name}') could not be read — it may be corrupted or in an unsupported format. It will be skipped.")
            continue
        valid_images.append(img)
        valid_filenames.append(f.name)
        with thumb_cols[i % len(thumb_cols)]:
            st.image(img, use_container_width=True)
            st.markdown(f'<div class="photo-thumb-label">Photo {i + 1}</div>', unsafe_allow_html=True)

# ---------- Farmer Information Form ----------
farmer_info = None
submitted = False

if valid_images:
    st.markdown('<div class="section-header">🌱 Crop Information</div>', unsafe_allow_html=True)
    st.caption("This helps interpret the AI result — it does not replace the image analysis itself.")

    with st.form("farmer_info_form"):
        fc1, fc2 = st.columns(2)
        with fc1:
            crop_name = st.selectbox("Crop name", ["Tomato", "Potato", "Bell Pepper", "Other / Not sure"])
            growth_stage = st.selectbox("Crop growth stage", ["Seedling", "Vegetative", "Flowering", "Fruiting", "Mature", "Not sure"])
            age_not_sure = st.checkbox("Not sure about crop age")
            crop_age_days = st.number_input("Approximate crop age (days)", min_value=1, max_value=365, value=30, step=1, disabled=age_not_sure)
        with fc2:
            symptom_duration = st.selectbox("How long have you noticed the symptoms?",
                                             ["Less than 1 day", "1–3 days", "4–7 days", "1–2 weeks", "More than 2 weeks", "Not sure"])
            affected_area = st.selectbox("How much of the crop appears affected?",
                                          ["Only one/few leaves", "Less than 25%", "25–50%", "50–75%", "More than 75%", "Not sure"])
            field_condition = st.selectbox("Recent weather / field condition",
                                            ["Normal", "High rainfall", "High humidity", "Very hot", "Very dry", "Not sure"])

        fc3, fc4 = st.columns(2)
        with fc3:
            prior_treatment = st.radio("Have you already applied any treatment?", ["No", "Yes"], horizontal=True)
            treatment_used = ""
            if prior_treatment == "Yes":
                treatment_used = st.text_input("Please specify the treatment used")
            previously_affected = st.selectbox("Has this crop shown this disease before? (optional)",
                                                ["Not sure", "No, first time", "Yes, it had this issue before"])
        with fc4:
            village_city = st.text_input("Village / City (optional)")
            district = st.text_input("District (optional)")

        submitted = st.form_submit_button("🔍 Analyze All Photos", use_container_width=True)

    if submitted:
        farmer_info = {
            "crop_name": crop_name,
            "growth_stage": growth_stage,
            "crop_age_days": None if age_not_sure else int(crop_age_days),
            "symptom_duration": symptom_duration,
            "affected_area": affected_area,
            "field_condition": field_condition,
            "prior_treatment": prior_treatment,
            "treatment_used": treatment_used,
            "previously_affected": previously_affected,
            "village_city": village_city,
            "district": district,
        }
        st.session_state.farmer_info = farmer_info

        if model is None:
            st.error("⚠️ The detection model isn't available right now, so photos couldn't be analyzed. Please try again later.")
        else:
            with st.spinner(f"Analyzing {len(valid_images)} photo(s)..."):
                results = analyze_multiple_images(valid_images, valid_filenames, model, class_names, farmer_info)
            st.session_state.analysis_results = results
            st.session_state.overall_assessment = calculate_overall_assessment(results)
else:
    st.markdown("""
        <div style="text-align:center; color:#9fab8f; padding: 2rem 1rem;">
            <div style="font-size:3.2rem;">🌱</div>
            <div style="margin-top:10px;">Upload one or more photos above to begin analysis</div>
        </div>
    """, unsafe_allow_html=True)

# ---------- Results ----------
if st.session_state.analysis_results:
    results = st.session_state.analysis_results
    overall = st.session_state.overall_assessment
    finfo = st.session_state.farmer_info

    lang_map = {
        "English": ("en", "treatment_en"),
        "हिंदी (Hindi)": ("hi", "treatment_hi"),
        "मराठी (Marathi)": ("mr", "treatment_mr"),
        "ಕನ್ನಡ (Kannada)": ("kn", "treatment_kn"),
    }
    lang_choice = st.selectbox("🌐 Voice / advisory language", list(lang_map.keys()))

    st.caption("AI predictions are advisory and should be verified with an agricultural expert when necessary. "
               "This tool does not claim 100% accuracy, and severity/progression figures are AI-based estimates, "
               "not scientifically validated measurements.")

    # ---- Individual results ----
    st.markdown('<div class="section-header">🔬 Individual Photo Results</div>', unsafe_allow_html=True)
    for entry in results:
        display_image_result(entry, lang_choice, lang_map)

    # ---- Overall assessment ----
    st.markdown('<div class="section-header">🌾 Overall Crop Health Assessment</div>', unsafe_allow_html=True)
    display_overall_dashboard(overall)

    # ---- Farmer information recap ----
    st.markdown('<div class="section-header">👨‍🌾 Farmer Information</div>', unsafe_allow_html=True)
    if finfo:
        age_display = f"{finfo['crop_age_days']} days" if finfo["crop_age_days"] else "Not sure"
        location_display = ", ".join([p for p in [finfo["village_city"], finfo["district"]] if p]) or "Not provided"
        treatment_display = finfo["treatment_used"] if finfo["prior_treatment"] == "Yes" and finfo["treatment_used"] else \
                             ("Yes (not specified)" if finfo["prior_treatment"] == "Yes" else "No")
        st.markdown(f"""
            <div class="info-card">
                <p><b>Crop:</b> {finfo['crop_name']}</p>
                <p><b>Growth stage:</b> {finfo['growth_stage']}</p>
                <p><b>Crop age:</b> {age_display}</p>
                <p><b>Symptoms noticed for:</b> {finfo['symptom_duration']}</p>
                <p><b>Affected area:</b> {finfo['affected_area']}</p>
                <p><b>Field condition:</b> {finfo['field_condition']}</p>
                <p><b>Previous treatment:</b> {treatment_display}</p>
                <p><b>Previously affected before:</b> {finfo['previously_affected']}</p>
                <p><b>Location:</b> {location_display}</p>
                <p style="font-size:0.78rem; color:#9fab8f; margin-top:8px;">This information helps put the AI
                   result in context (for example, how long symptoms have been present, or how much of the crop
                   is affected) — it does not by itself prove or confirm a diagnosis. The image-based AI
                   prediction remains the primary basis for the result above.</p>
            </div>
        """, unsafe_allow_html=True)

    # ---- Detailed recommendation based on the overall (most common) condition ----
    if overall.get("valid_total", 0) > 0 and finfo:
        st.markdown('<div class="section-header">🌱 Detailed Recommendation</div>', unsafe_allow_html=True)

        overall_is_healthy = overall["affected_count"] == 0
        crop_label = finfo["crop_name"] if finfo["crop_name"] != "Other / Not sure" else (overall["unique_crops"][0] if overall["unique_crops"] else "your crop")

        # growth stage: prefer farmer's explicit answer; fall back to age-based estimate
        if finfo["growth_stage"] != "Not sure":
            stage_name = finfo["growth_stage"]
            stage_note = "Based on the growth stage you selected."
        elif finfo["crop_age_days"]:
            stage_info = get_growth_stage(crop_label, finfo["crop_age_days"])
            stage_name = stage_info["stage"]
            stage_note = ("⚠️ Crop-specific growth-stage data isn't available for this crop — this is an "
                          "approximate estimate based on general growth patterns." if stage_info["is_estimate"]
                          else "Estimated from the crop age you provided; actual stage may vary with variety and local conditions.")
        else:
            stage_name = "Not determined"
            stage_note = "Growth stage could not be estimated — no stage or crop age was provided."

        st.markdown(f"""
            <div class="info-card">
                <h4>📅 Growth Stage</h4>
                <p><b>Crop:</b> {crop_label} &nbsp;|&nbsp; <b>Growth stage:</b> {stage_name}</p>
                <p style="font-size:0.78rem; color:#9fab8f;">{stage_note}</p>
            </div>
        """, unsafe_allow_html=True)

        severity_level_str = overall["overall_level"]
        overall_key = overall.get("overall_result_key")
        non_chemical_first = (not overall_is_healthy) and (
            severity_level_str == "Mild" or any(k in stage_name.lower() for k in ["seedling", "early", "sprouting"])
        )

        st.markdown('<div class="subsection-title">✅ Recommended Action</div>', unsafe_allow_html=True)
        if overall_is_healthy:
            st.markdown(f"""
                <div class="info-card">
                    <p>No disease was detected across the analyzed photos of this {crop_label.lower()} crop.</p>
                    <p>At the {stage_name.lower() if stage_name != "Not determined" else "current"} stage, continue
                       regular field monitoring, maintain good field hygiene, and re-check every few days, especially
                       during humid weather.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            css_class = _severity_details(severity_level_str)["css_class"] if severity_level_str else ""
            chem_line = (
                "Not urgently required yet — prioritize non-chemical/cultural control first "
                "(sanitation, spacing, avoiding leaf wetness) and monitor closely."
                if non_chemical_first else
                "A fungicide/bactericide treatment is appropriate at this stage if symptoms persist "
                "or worsen — see the Treatment section below."
            )
            next_check = "2–3 days" if severity_level_str == "Severe" else "3–5 days"
            st.markdown(f"""
                <div class="info-card {css_class}">
                    <p><b>1. Most frequently detected condition:</b> {overall['most_common_disease']}
                       ({overall['most_common_count']}/{overall['valid_total']} photos)</p>
                    <p><b>2. Overall estimated severity:</b> {severity_level_str}</p>
                    <p><b>3. Immediate step:</b> Remove and safely destroy visibly infected leaves/plant parts,
                       and avoid overhead irrigation to slow spread.</p>
                    <p><b>4. Chemical treatment needed?</b> {chem_line}</p>
                    <p><b>5. Next action:</b> Re-inspect the crop within {next_check} to track progress.</p>
                </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="subsection-title">💊 Treatment</div>', unsafe_allow_html=True)
        if overall_is_healthy:
            st.info("No treatment is needed right now — the analyzed photos show a healthy crop.")
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

            pesticide = get_pesticide_recommendation(overall_key) if overall_key else None
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

        st.markdown('<div class="subsection-title">⚠️ Growth-Stage Precaution</div>', unsafe_allow_html=True)
        if stage_name != "Not determined":
            st.info(get_stage_warning(crop_label, stage_name))
        else:
            st.caption("Provide a growth stage or crop age above to see a stage-specific precaution.")

        st.markdown('<div class="subsection-title">🌦️ Weather Risk</div>', unsafe_allow_html=True)
        weather_level, weather_message = get_weather_risk_from_condition(finfo["field_condition"])
        st.markdown(f"""
            <div class="info-card">
                <p><b>Reported field condition:</b> {finfo['field_condition']}</p>
                <p><b>Disease-spread risk:</b> {weather_level or 'Not estimated'}</p>
                <p>{weather_message}</p>
                <p style="font-size:0.78rem; color:#9fab8f;">Weather conditions alone do not confirm
                   disease presence — this only indicates how favorable conditions currently are for
                   disease spread.</p>
            </div>
        """, unsafe_allow_html=True)

        if finfo["previously_affected"] == "Yes, it had this issue before":
            st.caption("ℹ️ Since this field has had this issue before, disease pressure may already be "
                       "established — consider crop rotation and field sanitation for future seasons.")

        st.markdown('<div class="subsection-title">⏱️ How Quickly Should You Act?</div>', unsafe_allow_html=True)
        urgency = get_urgency(
            is_healthy=overall_is_healthy,
            severity_level=severity_level_str,
            stage_name=stage_name if stage_name != "Not determined" else "",
            weather_risk_level=weather_level,
        )
        urgency_color = {"🔴": "#e0665a", "🟠": "#f2a444", "🟡": "#f2c744", "🟢": "#7bd389"}[urgency["emoji"]]
        st.markdown(f"""
            <span class="urgency-badge" style="background-color:{urgency_color}22; color:{urgency_color}; border:1px solid {urgency_color};">
                {urgency['emoji']} {urgency['label']}
            </span>
            <p style="color:#d3ddc7; margin-top:6px;">{urgency['detail']}</p>
        """, unsafe_allow_html=True)

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
