import json
from pathlib import Path
from textwrap import dedent

import numpy as np
import streamlit as st
from PIL import Image


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="KrishiRakshak AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# FILE PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "crop_model.h5"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* =====================================================
       GLOBAL
       ===================================================== */

    .stApp {
        background: #f5f7f2;
        color: #17231a;
    }

    .main .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .stApp p {
        color: #3d493f !important;
    }

    .stApp label {
        color: #26352a !important;
        font-weight: 600 !important;
    }


    /* =====================================================
       HERO
       ===================================================== */

    .hero {
        background:
            linear-gradient(
                135deg,
                #173d2b 0%,
                #21563a 60%,
                #2f6b46 100%
            );

        padding: 2.8rem 3rem;
        border-radius: 26px;
        margin-bottom: 1.8rem;

        box-shadow:
            0 12px 35px rgba(23, 61, 43, 0.18);

        position: relative;
        overflow: hidden;
    }

    .hero::after {
        content: "🌾";
        position: absolute;
        right: 35px;
        bottom: -25px;
        font-size: 9rem;
        opacity: 0.12;
    }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 850;
        color: #ffffff !important;
        margin-bottom: 8px;
        position: relative;
        z-index: 2;
    }

    .hero-sub {
        font-size: 1.08rem;
        color: #dce9df !important;
        max-width: 700px;
        line-height: 1.6;
        position: relative;
        z-index: 2;
    }

    .hero-badge {
        display: inline-block;

        margin-top: 18px;
        padding: 7px 16px;

        border-radius: 30px;

        background: #f4c542;
        color: #173d2b !important;

        font-size: 0.78rem;
        font-weight: 800;

        position: relative;
        z-index: 2;
    }


    /* =====================================================
       SECTION TITLES
       ===================================================== */

    .section-header {
        font-size: 1.45rem;
        font-weight: 800;
        color: #173d2b !important;

        margin-top: 2.2rem;
        margin-bottom: 1rem;
    }


    /* =====================================================
       STAT CARDS
       ===================================================== */

    .stat-card {
        background: #ffffff;

        border: 1px solid #e1e8df;
        border-radius: 18px;

        padding: 1.25rem;

        text-align: center;

        min-height: 105px;

        box-shadow:
            0 5px 18px rgba(35, 63, 45, 0.07);
    }

    .stat-num {
        color: #17633b !important;

        font-size: 1.8rem;
        font-weight: 850;
    }

    .stat-label {
        color: #68746b !important;

        font-size: 0.82rem;
        margin-top: 4px;
    }


    /* =====================================================
       HOW IT WORKS
       ===================================================== */

    .step-card {
        background: #ffffff;

        border: 1px solid #e1e8df;
        border-radius: 18px;

        padding: 1.35rem 1.2rem;

        min-height: 155px;

        text-align: center;

        box-shadow:
            0 5px 18px rgba(35, 63, 45, 0.06);
    }

    .step-card b {
        color: #173d2b !important;
        font-size: 1rem;
    }

    .step-card span {
        color: #66736a !important;
        font-size: 0.86rem;
        line-height: 1.5;
    }

    .step-num {
        width: 38px;
        height: 38px;

        border-radius: 50%;

        background: #f4c542;
        color: #173d2b !important;

        display: flex;
        align-items: center;
        justify-content: center;

        font-weight: 850;

        margin: 0 auto 12px;
    }


    /* =====================================================
       SCANNER
       ===================================================== */

    .scanner-panel {
        background: #ffffff;

        border: 1px solid #dfe7df;

        border-radius: 22px;

        padding: 1.5rem;

        box-shadow:
            0 7px 24px rgba(35, 63, 45, 0.07);
    }

    .upload-panel-label {
        color: #173d2b !important;

        font-size: 1rem;
        font-weight: 750;

        margin-bottom: 0.7rem;
    }

    div[data-testid="stFileUploader"] {
        background: #ffffff !important;
    }

    div[data-testid="stFileUploaderDropzone"] {
        background: #f8faf7 !important;

        border: 2px dashed #9db9a4 !important;

        border-radius: 17px !important;

        min-height: 145px;

        padding: 1rem !important;
    }

    div[data-testid="stFileUploaderDropzone"]:hover {
        border-color: #217346 !important;

        background: #f2f8f3 !important;
    }

    div[data-testid="stFileUploaderDropzone"] span,
    div[data-testid="stFileUploaderDropzone"] small {
        color: #405047 !important;
    }

    div[data-testid="stFileUploaderDropzone"] button {
        background: #17633b !important;

        color: #ffffff !important;

        border: none !important;

        border-radius: 10px !important;

        font-weight: 700 !important;
    }


    /* =====================================================
       IMAGE
       ===================================================== */

    [data-testid="stImage"] {
        border-radius: 16px;
        overflow: hidden;
    }


    /* =====================================================
       RESULT CARD
       ===================================================== */

    .result-card {
        background: #ffffff;

        border: 1px solid #dfe7df;

        border-radius: 20px;

        padding: 1.6rem 1.7rem;

        box-shadow:
            0 7px 24px rgba(35, 63, 45, 0.08);
    }

    .result-label {
        color: #758178 !important;

        font-size: 0.73rem;

        font-weight: 800;

        letter-spacing: 1.3px;

        text-transform: uppercase;
    }

    .result-name {
        color: #173d2b !important;

        font-size: 1.65rem;

        font-weight: 850;

        margin-top: 7px;
        margin-bottom: 8px;
    }

    .result-icon {
        font-size: 2.4rem;
        margin-bottom: 4px;
    }


    /* =====================================================
       CONFIDENCE
       ===================================================== */

    .confidence-bg {
        width: 100%;

        height: 12px;

        background: #e9eee9;

        border-radius: 20px;

        overflow: hidden;

        margin-top: 8px;
        margin-bottom: 8px;
    }

    .confidence-fill {
        height: 12px;

        border-radius: 20px;
    }

    .confidence-text {
        color: #66736a !important;

        font-size: 0.85rem;

        font-weight: 700;
    }


    /* =====================================================
       SEVERITY
       ===================================================== */

    .severity-badge {
        display: inline-block;

        padding: 7px 14px;

        border-radius: 30px;

        font-size: 0.78rem;

        font-weight: 800;

        margin-top: 8px;
    }


    /* =====================================================
       TREATMENT
       ===================================================== */

    .treatment-box {
        background: #fff9e6;

        border: 1px solid #f0d879;

        border-radius: 17px;

        padding: 1.25rem 1.4rem;

        margin-top: 1.2rem;
    }

    .treatment-title {
        color: #8a6810 !important;

        font-weight: 800;

        margin-bottom: 7px;
    }

    .treatment-text {
        color: #4c4b3e !important;

        line-height: 1.6;
    }


    /* =====================================================
       SELECTBOX
       ===================================================== */

    div[data-baseweb="select"] > div {
        background: #ffffff !important;

        border: 1px solid #ccd8ce !important;

        border-radius: 11px !important;

        color: #26352a !important;
    }

    div[data-baseweb="select"] span {
        color: #26352a !important;
    }


    /* =====================================================
       BUTTONS
       ===================================================== */

    .stButton > button {
        background: #17633b !important;

        color: #ffffff !important;

        border: none !important;

        border-radius: 11px !important;

        font-weight: 750 !important;

        min-height: 44px;

        transition: 0.2s ease;
    }

    .stButton > button:hover {
        background: #0f4e2d !important;

        transform: translateY(-1px);
    }


    /* =====================================================
       HELPLINE
       ===================================================== */

    .helpline-card {
        background: #ffffff;

        border: 1px solid #dfe7df;

        border-radius: 19px;

        padding: 1.4rem 1.6rem;

        box-shadow:
            0 5px 18px rgba(35, 63, 45, 0.06);
    }

    .helpline-title {
        color: #17633b !important;

        font-weight: 800;
        font-size: 1.05rem;
    }

    .helpline-text {
        color: #536057 !important;

        font-size: 0.92rem;
        line-height: 1.6;
    }


    /* =====================================================
       FUTURE CROPS
       ===================================================== */

    .roadmap-card {
        background: #ffffff;

        border: 1px solid #dfe7df;

        border-radius: 17px;

        padding: 1.15rem;

        min-height: 120px;

        box-shadow:
            0 5px 16px rgba(35, 63, 45, 0.05);
    }

    .roadmap-title {
        color: #173d2b !important;

        font-weight: 800;
    }

    .roadmap-text {
        color: #68746b !important;

        font-size: 0.82rem;
        line-height: 1.5;
    }


    /* =====================================================
       FOOTER
       ===================================================== */

    .footer-note {
        color: #7a857d !important;

        font-size: 0.78rem;

        text-align: center;

        margin-top: 3rem;

        padding-top: 1.3rem;

        border-top: 1px solid #dce4dc;
    }


    /* =====================================================
       ALERTS
       ===================================================== */

    .stAlert {
        border-radius: 12px !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# DISEASE INFORMATION
# =========================================================

DISEASE_INFO = {

    "Pepper__bell___Bacterial_spot": {
        "display": "Bell Pepper — Bacterial Spot",
        "icon": "🫑",

        "treatment_en": (
            "Use a suitable copper-based bactericide according "
            "to the product label. Avoid overhead watering and "
            "remove severely infected leaves."
        ),

        "treatment_hi": (
            "उत्पाद के लेबल के अनुसार उपयुक्त कॉपर आधारित "
            "बैक्टीरिसाइड का प्रयोग करें। ऊपर से पानी देने से "
            "बचें और अधिक संक्रमित पत्तियों को हटा दें।"
        ),

        "treatment_mr": (
            "उत्पादनाच्या लेबलनुसार योग्य तांबेयुक्त "
            "बॅक्टेरिसाइड वापरा. वरून पाणी देणे टाळा आणि "
            "जास्त संक्रमित पाने काढून टाका."
        ),

        "treatment_kn": (
            "ಉತ್ಪನ್ನದ ಲೇಬಲ್ ಪ್ರಕಾರ ಸೂಕ್ತವಾದ ತಾಮ್ರ ಆಧಾರಿತ "
            "ಬ್ಯಾಕ್ಟೀರಿಸೈಡ್ ಬಳಸಿ. ಮೇಲಿನಿಂದ ನೀರುಣಿಸುವುದನ್ನು "
            "ತಪ್ಪಿಸಿ ಮತ್ತು ಹೆಚ್ಚು ಸೋಂಕಿತ ಎಲೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ."
        ),

        "severity": "moderate",
    },


    "Potato___Early_blight": {
        "display": "Potato — Early Blight",
        "icon": "🥔",

        "treatment_en": (
            "Use an appropriate fungicide according to the "
            "product label. Remove infected plant debris and "
            "maintain good field hygiene."
        ),

        "treatment_hi": (
            "उत्पाद के लेबल के अनुसार उपयुक्त फफूंदनाशक का "
            "प्रयोग करें। संक्रमित पौधों के अवशेष हटाएं और "
            "खेत की स्वच्छता बनाए रखें।"
        ),

        "treatment_mr": (
            "उत्पादनाच्या लेबलनुसार योग्य बुरशीनाशक वापरा. "
            "संक्रमित वनस्पतींचे अवशेष काढून टाका आणि "
            "शेताची स्वच्छता राखा."
        ),

        "treatment_kn": (
            "ಉತ್ಪನ್ನದ ಲೇಬಲ್ ಪ್ರಕಾರ ಸೂಕ್ತವಾದ ಶಿಲೀಂಧ್ರನಾಶಕ ಬಳಸಿ. "
            "ಸೋಂಕಿತ ಸಸ್ಯದ ಅವಶೇಷಗಳನ್ನು ತೆಗೆದುಹಾಕಿ ಮತ್ತು "
            "ಹೊಲದ ಸ್ವಚ್ಛತೆಯನ್ನು ಕಾಪಾಡಿ."
        ),

        "severity": "moderate",
    },


    "Tomato_Late_blight": {
        "display": "Tomato — Late Blight",
        "icon": "🍅",

        "treatment_en": (
            "Use an appropriate fungicide according to the "
            "product label and remove severely infected plant "
            "material to help reduce disease spread."
        ),

        "treatment_hi": (
            "उत्पाद के लेबल के अनुसार उपयुक्त फफूंदनाशक का "
            "प्रयोग करें और रोग के फैलाव को कम करने के लिए "
            "अधिक संक्रमित पौधों के हिस्सों को हटा दें।"
        ),

        "treatment_mr": (
            "उत्पादनाच्या लेबलनुसार योग्य बुरशीनाशक वापरा आणि "
            "रोगाचा प्रसार कमी करण्यासाठी जास्त संक्रमित "
            "भाग काढून टाका."
        ),

        "treatment_kn": (
            "ಉತ್ಪನ್ನದ ಲೇಬಲ್ ಪ್ರಕಾರ ಸೂಕ್ತವಾದ ಶಿಲೀಂಧ್ರನಾಶಕ ಬಳಸಿ "
            "ಮತ್ತು ರೋಗ ಹರಡುವಿಕೆಯನ್ನು ಕಡಿಮೆ ಮಾಡಲು ಹೆಚ್ಚು "
            "ಸೋಂಕಿತ ಭಾಗಗಳನ್ನು ತೆಗೆದುಹಾಕಿ."
        ),

        "severity": "severe",
    },


    "Tomato_healthy": {
        "display": "Tomato — Healthy",
        "icon": "✅",

        "treatment_en": (
            "No disease detected. Continue regular monitoring "
            "and maintain good field hygiene."
        ),

        "treatment_hi": (
            "कोई रोग नहीं पाया गया। नियमित निगरानी और अच्छी "
            "खेत स्वच्छता जारी रखें।"
        ),

        "treatment_mr": (
            "कोणताही रोग आढळला नाही. नियमित देखरेख आणि "
            "चांगली शेत स्वच्छता सुरू ठेवा."
        ),

        "treatment_kn": (
            "ಯಾವುದೇ ರೋಗ ಪತ್ತೆಯಾಗಿಲ್ಲ. ನಿಯಮಿತ ಮೇಲ್ವಿಚಾರಣೆ "
            "ಮತ್ತು ಉತ್ತಮ ಹೊಲದ ನೈರ್ಮಲ್ಯವನ್ನು ಮುಂದುವರಿಸಿ."
        ),

        "severity": "healthy",
    },
}


# =========================================================
# SEVERITY SETTINGS
# =========================================================

SEVERITY_COLORS = {
    "healthy": "#2e8b57",
    "moderate": "#d49b16",
    "severe": "#d9534f",
}

SEVERITY_LABELS = {
    "healthy": "Healthy",
    "moderate": "Moderate Risk",
    "severe": "Severe — Act Now",
}


# =========================================================
# LANGUAGE SETTINGS
# =========================================================

LANGUAGES = {
    "English": "treatment_en",
    "हिंदी (Hindi)": "treatment_hi",
    "मराठी (Marathi)": "treatment_mr",
    "ಕನ್ನಡ (Kannada)": "treatment_kn",
}


# =========================================================
# MODEL LOADING
# =========================================================

@st.cache_resource
def load_my_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH.name} was not found in the application folder."
        )

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"{CLASS_NAMES_PATH.name} was not found in the application folder."
        )

    try:
        from tensorflow.keras.models import load_model

        model = load_model(MODEL_PATH)

        with open(
            CLASS_NAMES_PATH,
            "r",
            encoding="utf-8",
        ) as file:
            class_names = json.load(file)

        if isinstance(class_names, dict):

            try:
                class_names = [
                    class_names[str(i)]
                    for i in range(len(class_names))
                ]

            except KeyError:
                class_names = list(class_names.values())

        if not isinstance(class_names, list):
            raise ValueError(
                "class_names.json must contain a list or dictionary."
            )

        return model, class_names

    except Exception as error:

        raise RuntimeError(
            f"Unable to load the AI model: {error}"
        ) from error


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_disease(model, image):

    image = image.convert("RGB")

    image = image.resize((224, 224))

    image_array = np.asarray(image).astype("float32")

    image_array = image_array / 255.0

    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(
        image_array,
        verbose=0,
    )

    prediction = np.asarray(prediction)

    # Handle binary-output models
    if prediction.ndim == 2 and prediction.shape[1] == 1:

        probability = float(prediction[0][0])

        if probability >= 0.5:
            class_index = 1
            confidence = probability
        else:
            class_index = 0
            confidence = 1.0 - probability

    else:

        probabilities = prediction[0]

        class_index = int(
            np.argmax(probabilities)
        )

        confidence = float(
            probabilities[class_index]
        )

    return class_index, confidence


# =========================================================
# HERO
# =========================================================

st.markdown(
    dedent(
        """
        <div class="hero">

            <div class="hero-title">
                🌾 KrishiRakshak AI
            </div>

            <div class="hero-sub">
                AI-powered crop disease detection from a single
                leaf image — enabling faster and smarter
                agricultural decision-making.
            </div>

            <div class="hero-badge">
                SIH 2026 · SIH26131 · Government of Maharashtra
            </div>

        </div>
        """
    ),
    unsafe_allow_html=True,
)


# =========================================================
# LOAD MODEL
# =========================================================

try:

    model, class_names = load_my_model()

except Exception as error:

    st.error("⚠️ AI model could not be loaded.")

    st.code(str(error))

    st.info(
        "Make sure crop_model.h5 and class_names.json "
        "are present in the same folder as app.py."
    )

    st.stop()


# =========================================================
# STATISTICS
# =========================================================

s1, s2, s3, s4 = st.columns(4)

with s1:

    st.markdown(
        dedent(
            """
            <div class="stat-card">
                <div class="stat-num">4</div>
                <div class="stat-label">
                    Disease Classes
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


with s2:

    st.markdown(
        dedent(
            """
            <div class="stat-card">
                <div class="stat-num">AI</div>
                <div class="stat-label">
