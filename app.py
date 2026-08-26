import json
from pathlib import Path

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

        min-height: 150px;

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
       UPLOAD
       ===================================================== */

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
    }


    /* =====================================================
       CONFIDENCE BAR
       ===================================================== */

    .confidence-bg {
        width: 100%;

        height: 12px;

        background: #e9eee9;

        border-radius: 20px;

        overflow: hidden;

        margin-top: 8px;
    }

    .confidence-fill {
        height: 12px;

        border-radius: 20px;
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
    }

    .helpline-text {
        color: #536057 !important;

        font-size: 0.92rem;
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
            encoding="utf-8"
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
# HERO
# =========================================================

st.markdown(
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
    """,
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
        """
        <div class="stat-card">
            <div class="stat-num">4</div>
            <div class="stat-label">
                Disease Classes
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with s2:

    st.markdown(
        """
        <div class="stat-card">
            <div class="stat-num">AI</div>
            <div class="stat-label">
                Image-Based Detection
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with s3:

    st.markdown(
        """
        <div class="stat-card">
            <div class="stat-num">4</div>
            <div class="stat-label">
                Languages
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with s4:

    st.markdown(
        """
        <div class="stat-card">
            <div class="stat-num">224×224</div>
            <div class="stat-label">
                Image Input
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# HOW IT WORKS
# =========================================================

st.markdown(
    '<div class="section-header">⚡ How it works</div>',
    unsafe_allow_html=True,
)

h1, h2, h3 = st.columns(3)


with h1:

    st.markdown(
        """
        <div class="step-card">

            <div class="step-num">1</div>

            <b>📷 Upload a Leaf</b>

            <br><br>

            <span>
                Upload a clear image of the
                affected crop leaf.
            </span>

        </div>
        """,
        unsafe_allow_html=True,
    )


with h2:

    st.markdown(
        """
        <div class="step-card">

            <div class="step-num">2</div>

            <b>🧠 AI Analysis</b>

            <br><br>

            <span>
                The trained AI model analyzes
                the image and predicts the condition.
            </span>

        </div>
        """,
        unsafe_allow_html=True,
    )


with h3:

    st.markdown(
        """
        <div class="step-card">

            <div class="step-num">3</div>

            <b>🗣️ Get Guidance</b>

            <br><br>

            <span>
                Receive disease information and
                treatment guidance in your language.
            </span>

        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# SCANNER
# =========================================================

st.markdown(
    '<div class="section-header">📸 Scan a Crop Leaf</div>',
    unsafe_allow_html=True,
)

col_upload, col_result = st.columns(
    [1, 1.2],
    gap="large",
)


# =========================================================
# UPLOAD COLUMN
# =========================================================

with col_upload:

    st.markdown(
        """
        <div class="upload-panel-label">
            Upload a clear crop leaf image
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload crop leaf image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        label_visibility="collapsed",
    )

    img = None

    if uploaded is not None:

        try:

            img = Image.open(uploaded).convert("RGB")

            st.image(
                img,
                caption="Uploaded crop image",
                use_container_width=True,
            )

        except Exception:

            st.error(
                "The uploaded file could not be read."
            )


# =========================================================
# RESULT COLUMN
# =========================================================

with col_result:

    if img is None:

        st.markdown(
            """
            <div class="result-card">

                <div
                    style="
                        text-align:center;
                        padding:3rem 1rem;
                    "
                >

                    <div
                        style="
                            font-size:3.5rem;
                        "
                    >
                        🌱
                    </div>

                    <div
                        style="
                            color:#173d2b;
                            font-size:1.1rem;
                            font-weight:750;
                            margin-top:10px;
                        "
                    >
                        Your AI result will appear here
                    </div>

                    <div
                        style="
                            color:#718078;
                            margin-top:7px;
                            font-size:0.9rem;
                        "
                    >
                        Upload a crop leaf image
                        to begin detection.
                    </div>

                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        try:

            # -------------------------------------------------
            # IMAGE PREPROCESSING
            # -------------------------------------------------

            resized_image = img.resize(
                (224, 224),
                Image.Resampling.LANCZOS,
            )

            image_array = np.asarray(
                resized_image,
                dtype=np.float32,
            )

            image_array = image_array / 255.0

            image_array = np.expand_dims(
                image_array,
                axis=0,
            )


            # -------------------------------------------------
            # PREDICTION
            # -------------------------------------------------

            prediction = model.predict(
                image_array,
                verbose=0,
            )

            prediction = np.asarray(prediction)

            if prediction.ndim == 2:
                prediction = prediction[0]

            if prediction.ndim != 1:
                raise ValueError(
                    "Unexpected model output shape."
                )

            predicted_index = int(
                np.argmax(prediction)
            )

            confidence = float(
                np.max(prediction)
            ) * 100.0


            # -------------------------------------------------
            # CLASS CHECK
            # -------------------------------------------------

            if predicted_index >= len(class_names):

                raise ValueError(
                    "The number of model outputs does not "
                    "match class_names.json."
                )

            result = class_names[predicted_index]


            # -------------------------------------------------
            # DISEASE INFORMATION
            # -------------------------------------------------

            info = DISEASE_INFO.get(
                result,
                {
                    "display": str(result).replace(
                        "_",
                        " "
                    ),

                    "icon": "🌿",

                    "treatment_en": (
                        "The model detected this condition. "
                        "Please consult a local agriculture "
                        "expert for confirmation and treatment."
                    ),

                    "treatment_hi": (
                        "मॉडल ने इस स्थिति का पता लगाया है। "
                        "पुष्टि और उपचार के लिए स्थानीय कृषि "
                        "विशेषज्ञ से सलाह लें।"
                    ),

                    "treatment_mr": (
                        "मॉडेलने ही स्थिती शोधली आहे. "
                        "पुष्टी आणि उपचारासाठी स्थानिक कृषी "
                        "तज्ञांचा सल्ला घ्या."
                    ),

                    "treatment_kn": (
                        "ಮಾದರಿಯು ಈ ಸ್ಥಿತಿಯನ್ನು ಪತ್ತೆಹಚ್ಚಿದೆ. "
                        "ದೃಢೀಕರಣ ಮತ್ತು ಚಿಕಿತ್ಸೆಗೆ ಸ್ಥಳೀಯ ಕೃಷಿ "
                        "ತಜ್ಞರನ್ನು ಸಂಪರ್ಕಿಸಿ."
                    ),

                    "severity": "moderate",
                },
            )


            severity = info.get(
                "severity",
                "moderate"
            )

            severity_color = SEVERITY_COLORS.get(
                severity,
                "#d49b16"
            )

            severity_label = SEVERITY_LABELS.get(
                severity,
                "Detected"
            )


            # -------------------------------------------------
            # RESULT CARD
            # -------------------------------------------------

            st.markdown(
                f"""
                <div
                    class="result-card"
                    style="
                        border-top:
                        5px solid {severity_color};
                    "
                >

                    <div class="result-label">
                        AI Detection Result
                    </div>

                    <div class="result-name">
                        {info["icon"]}
                        {info["display"]}
                    </div>

                    <div
                        style="
                            color:{severity_color};
                            font-weight:800;
                            margin-top:12px;
                        "
                    >
                        ● {severity_label}
                    </div>

                    <div
                        style="
                            color:#59665d;
                            margin-top:16px;
                            font-size:0.9rem;
                        "
                    >
                        Model Confidence:
                        <strong>
                            {confidence:.1f}%
                        </strong>
                    </div>

                    <div class="confidence-bg">

                        <div
                            class="confidence-fill"
                            style="
                                width:
                                {min(confidence, 100):.1f}%;

                                background:
                                {severity_color};
                            "
                        >
                        </div>

                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


            # -------------------------------------------------
            # LANGUAGE SELECTION
            # -------------------------------------------------

            lang_choice = st.selectbox(
                "🌐 Select language for guidance",
                list(LANGUAGES.keys()),
            )

            treatment_key = LANGUAGES[
                lang_choice
            ]

            treatment_text = info.get(
                treatment_key,
                info["treatment_en"],
            )


            # -------------------------------------------------
            # TREATMENT
            # -------------------------------------------------

            st.markdown(
                f"""
                <div class="treatment-box">

                    <div class="treatment-title">
                        💊 Recommended Action
                    </div>

                    <div class="treatment-text">
                        {treatment_text}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


            # -------------------------------------------------
            # VOICE GUIDANCE
            # -------------------------------------------------

            if st.button(
                "🔊 Play Voice Advice",
                use_container_width=True,
            ):

                try:

                    from gtts import gTTS

                    language_codes = {
                        "English": "en",
                        "हिंदी (Hindi)": "hi",
                        "मराठी (Marathi)": "mr",
                        "ಕನ್ನಡ (Kannada)": "kn",
                    }

                    lang_code = language_codes[
                        lang_choice
                    ]

                    audio_file = BASE_DIR / "output.mp3"

                    tts = gTTS(
                        text=(
                            f"{info['display']}. "
                            f"{treatment_text}"
                        ),
                        lang=lang_code,
                    )

                    tts.save(audio_file)

                    with open(
                        audio_file,
                        "rb"
                    ) as audio:

                        st.audio(
                            audio.read(),
                            format="audio/mp3",
                        )

                except Exception as error:

                    st.warning(
                        "Voice generation is unavailable "
                        "in the current deployment."
                    )

                    st.caption(
                        str(error)
                    )


        except Exception as error:

            st.error(
                "⚠️ Prediction failed."
            )

            st.code(
                str(error)
            )


# =========================================================
# FARMER SUPPORT
# =========================================================

st.markdown(
    '<div class="section-header">📞 Farmer Support</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="helpline-card">

        <div class="helpline-title">
            📱 Kisan Call Centre
        </div>

        <div class="helpline-text">
            Government of India farmer helpline
        </div>

        <div
            style="
                color:#173d2b;
                font-size:1.2rem;
                font-weight:800;
                margin-top:7px;
            "
        >
            1800-180-1551
        </div>

        <div class="helpline-text">
            Available in multiple Indian languages.
        </div>

        <hr>

        <div class="helpline-title">
            📱 PM-KISAN Helpline
        </div>

        <div class="helpline-text">

            155261 / 1800-115-526

            &nbsp;&nbsp;|&nbsp;&nbsp;

            011-24300606

        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# FUTURE EXPANSION
# =========================================================

st.markdown(
    '<div class="section-header">🌱 Future Expansion</div>',
    unsafe_allow_html=True,
)

st.write(
    "The current prototype focuses on selected crop disease "
    "classes. The platform can be expanded to Maharashtra's "
    "major agricultural crops."
)


r1, r2, r3, r4 = st.columns(4)


with r1:

    st.markdown(
        """
        <div class="roadmap-card">

            <div class="roadmap-title">
                🌾 Jowar
            </div>

            <div class="roadmap-text">
                Grain mold<br>
                Downy mildew
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


with r2:

    st.markdown(
        """
        <div class="roadmap-card">

            <div class="roadmap-title">
                🌾 Rice
            </div>

            <div class="roadmap-text">
                Blast<br>
                Bacterial blight
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


with r3:

    st.markdown(
        """
        <div class="roadmap-card">

            <div class="roadmap-title">
                🌿 Cotton
            </div>

            <div class="roadmap-text">
                Pink bollworm<br>
                Leaf curl
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


with r4:

    st.markdown(
        """
        <div class="roadmap-card">

            <div class="roadmap-title">
                🎋 Sugarcane
            </div>

            <div class="roadmap-text">
                Red rot<br>
                Smut
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# SIH VALUE PROPOSITION
# =========================================================

st.markdown(
    '<div class="section-header">🚀 Why KrishiRakshak AI?</div>',
    unsafe_allow_html=True,
)

v1, v2, v3 = st.columns(3)


with v1:

    st.markdown(
        """
        <div class="step-card">

            <div style="font-size:2rem;">
                📱
            </div>

            <b>Accessible</b>

            <br><br>

            <span>
                Simple image-based interaction
                designed for farmers.
            </span>

        </div>
        """,
        unsafe_allow_html=True,
    )


with v2:

    st.markdown(
        """
        <div class="step-card">

            <div style="font-size:2rem;">
                🌐
            </div>

            <b>Multilingual</b>

            <br><br>

            <span>
                Guidance can be presented
                in multiple Indian languages.
            </span>

        </div>
        """,
        unsafe_allow_html=True,
    )


with v3:

    st.markdown(
        """
        <div class="step-card">

            <div style="font-size:2rem;">
                🧠
            </div>

            <b>AI-Powered</b>

            <br><br>

            <span>
                Machine learning enables
                rapid preliminary disease detection.
            </span>

        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer-note">

        KrishiRakshak AI · SIH 2026 · Problem Statement SIH26131

        <br>

        AI-assisted crop disease detection prototype

    </div>
    """,
    unsafe_allow_html=True,
)
