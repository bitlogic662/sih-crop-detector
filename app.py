import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="KrishiRakshak AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "crop_model.h5"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    /* Main application background */
    .stApp {
        background:
            radial-gradient(
                circle at top left,
                #2b3d22 0%,
                #1c2a17 55%,
                #141f10 100%
            );
    }

    /* General text */
    .stApp p,
    .stApp span,
    .stApp label,
    .stApp li,
    .stMarkdown,
    .stCaption {
        color: #eef2e6 !important;
    }

    /* ---------------- HERO ---------------- */

    .hero {
        background:
            linear-gradient(
                135deg,
                #23331b 0%,
                #35492a 45%,
                #4a6339 100%
            );

        padding: 2.4rem 2.6rem;
        border-radius: 24px;
        margin-bottom: 1.8rem;

        box-shadow:
            0 12px 32px rgba(0, 0, 0, 0.35);

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

    /* ---------------- STAT CARDS ---------------- */

    .stat-card {
        background: rgba(255,255,255,0.06);
        border-radius: 18px;
        padding: 1.1rem 1.2rem;
        text-align: center;

        box-shadow:
            0 4px 16px rgba(0,0,0,0.25);

        border: 1px solid rgba(255,255,255,0.10);
        color: #f6f9f2;
        height: 100%;
    }

    .stat-num {
        font-size: 1.8rem;
        font-weight: 800;
        color: #f2c744;
    }

    .stat-label {
        font-size: 0.8rem;
        color: #d3ddc7;
        margin-top: 2px;
    }

    /* ---------------- SECTION HEADERS ---------------- */

    .section-header {
        font-size: 1.35rem;
        font-weight: 800;
        color: #f6f9f2;
        margin: 2.2rem 0 1rem 0;
    }

    /* ---------------- STEP CARDS ---------------- */

    .step-card {
        background: rgba(255,255,255,0.06);
        color: #f6f9f2;

        border-radius: 18px;
        padding: 1.3rem 1.2rem;

        text-align: center;

        border: 1px solid rgba(255,255,255,0.10);

        box-shadow:
            0 4px 14px rgba(0,0,0,0.22);

        height: 100%;
    }

    .step-num {
        width: 34px;
        height: 34px;

        border-radius: 50%;

        background: #f2c744;
        color: #23331b;

        display: flex;
        align-items: center;
        justify-content: center;

        font-weight: 800;

        margin: 0 auto 10px auto;
    }

    /* ---------------- UPLOAD ---------------- */

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

        box-shadow:
            0 3px 14px rgba(0,0,0,0.2);
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

    /* ---------------- RESULT ---------------- */

    .result-card {
        background: rgba(255,255,255,0.06);

        border-radius: 20px;

        padding: 1.6rem 1.8rem;

        box-shadow:
            0 8px 26px rgba(0,0,0,0.3);

        border: 1px solid rgba(255,255,255,0.10);

        border-top: 5px solid #f2c744;
    }

    .result-label {
        font-size: 0.78rem;
        font-weight: 700;

        letter-spacing: 1.2px;

        color: #bcc7ab;

        text-transform: uppercase;
    }

    .result-name {
        font-size: 1.7rem;
        font-weight: 800;

        margin-top: 4px;

        color: #f6f9f2;
    }

    .status-dot {
        display: inline-block;

        width: 10px;
        height: 10px;

        border-radius: 50%;

        margin-right: 6px;
    }

    /* ---------------- CONFIDENCE BAR ---------------- */

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
    }

    /* ---------------- TREATMENT ---------------- */

    .treatment-box {
        background: rgba(242,199,68,0.10);

        border-radius: 18px;

        padding: 1.3rem 1.6rem;

        margin-top: 1.2rem;

        border: 1px solid rgba(242,199,68,0.35);
    }

    /* ---------------- HELPLINE ---------------- */

    .helpline-card {
        background: rgba(255,255,255,0.06);

        border-radius: 18px;

        padding: 1.4rem 1.7rem;

        border: 1px solid rgba(255,255,255,0.10);
    }

    /* ---------------- ROADMAP ---------------- */

    .roadmap-chip {
        background: rgba(255,255,255,0.06);

        color: #f6f9f2;

        border-radius: 16px;

        padding: 1rem 1.1rem;

        margin-bottom: 8px;

        border: 1px solid rgba(255,255,255,0.10);

        box-shadow:
            0 3px 10px rgba(0,0,0,0.2);

        height: 100%;
    }

    .roadmap-chip span {
        color: #d3ddc7 !important;
    }

    /* ---------------- BUTTON ---------------- */

    .stButton button {
        background: #f2c744 !important;

        color: #23331b !important;

        border: none !important;

        border-radius: 30px !important;

        font-weight: 700 !important;

        padding: 0.6rem 1.2rem !important;
    }

    .stButton button:hover {
        background: #f6d768 !important;
    }

    /* ---------------- FOOTER ---------------- */

    .footer-note {
        color: #9fab8f;

        font-size: 0.8rem;

        margin-top: 3rem;

        text-align: center;

        padding-top: 1.5rem;

        border-top:
            1px solid rgba(255,255,255,0.10);
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# DISEASE INFORMATION
# ---------------------------------------------------------

DISEASE_INFO = {
    "Pepper__bell___Bacterial_spot": {
        "display": "Bell Pepper — Bacterial Spot",
        "icon": "🫑",
        "treatment_en": (
            "Apply a suitable copper-based bactericide according to "
            "the product label. Avoid overhead watering and remove "
            "severely infected leaves."
        ),
        "treatment_hi": (
            "उत्पाद के लेबल के अनुसार उपयुक्त कॉपर आधारित बैक्टीरिसाइड "
            "का प्रयोग करें। ऊपर से पानी देने से बचें और अधिक संक्रमित "
            "पत्तियों को हटा दें।"
        ),
        "treatment_mr": (
            "उत्पादनाच्या लेबलनुसार योग्य तांबेयुक्त बॅक्टेरिसाइड वापरा. "
            "वरून पाणी देणे टाळा आणि जास्त संक्रमित पाने काढून टाका."
        ),
        "treatment_kn": (
            "ಉತ್ಪನ್ನದ ಲೇಬಲ್ ಪ್ರಕಾರ ಸೂಕ್ತವಾದ ತಾಮ್ರ ಆಧಾರಿತ ಬ್ಯಾಕ್ಟೀರಿಸೈಡ್ "
            "ಬಳಸಿ. ಮೇಲಿನಿಂದ ನೀರುಣಿಸುವುದನ್ನು ತಪ್ಪಿಸಿ ಮತ್ತು ಹೆಚ್ಚು ಸೋಂಕಿತ "
            "ಎಲೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ."
        ),
        "severity": "moderate",
    },

    "Potato___Early_blight": {
        "display": "Potato — Early Blight",
        "icon": "🥔",
        "treatment_en": (
            "Use an appropriate fungicide according to the product "
            "label. Remove infected plant debris and maintain good "
            "field hygiene."
        ),
        "treatment_hi": (
            "उत्पाद के लेबल के अनुसार उपयुक्त फफूंदनाशक का प्रयोग करें। "
            "संक्रमित पौधों के अवशेष हटाएं और खेत की स्वच्छता बनाए रखें।"
        ),
        "treatment_mr": (
            "उत्पादनाच्या लेबलनुसार योग्य बुरशीनाशक वापरा. संक्रमित "
            "वनस्पतींचे अवशेष काढून टाका आणि शेताची स्वच्छता राखा."
        ),
        "treatment_kn": (
            "ಉತ್ಪನ್ನದ ಲೇಬಲ್ ಪ್ರಕಾರ ಸೂಕ್ತವಾದ ಶಿಲೀಂಧ್ರನಾಶಕ ಬಳಸಿ. "
            "ಸೋಂಕಿತ ಸಸ್ಯದ ಅವಶೇಷಗಳನ್ನು ತೆಗೆದುಹಾಕಿ ಮತ್ತು ಹೊಲದ ಸ್ವಚ್ಛತೆಯನ್ನು ಕಾಪಾಡಿ."
        ),
        "severity": "moderate",
    },

    "Tomato_Late_blight": {
        "display": "Tomato — Late Blight",
        "icon": "🍅",
        "treatment_en": (
            "Use an appropriate fungicide according to the product "
            "label and remove severely infected plant material to "
            "help reduce disease spread."
        ),
        "treatment_hi": (
            "उत्पाद के लेबल के अनुसार उपयुक्त फफूंदनाशक का प्रयोग करें "
            "और रोग के फैलाव को कम करने के लिए अधिक संक्रमित पौधों के "
            "हिस्सों को हटा दें।"
        ),
        "treatment_mr": (
            "उत्पादनाच्या लेबलनुसार योग्य बुरशीनाशक वापरा आणि रोगाचा "
            "प्रसार कमी करण्यासाठी जास्त संक्रमित भाग काढून टाका."
        ),
        "treatment_kn": (
            "ಉತ್ಪನ್ನದ ಲೇಬಲ್ ಪ್ರಕಾರ ಸೂಕ್ತವಾದ ಶಿಲೀಂಧ್ರನಾಶಕ ಬಳಸಿ ಮತ್ತು "
            "ರೋಗ ಹರಡುವಿಕೆಯನ್ನು ಕಡಿಮೆ ಮಾಡಲು ಹೆಚ್ಚು ಸೋಂಕಿತ ಭಾಗಗಳನ್ನು ತೆಗೆದುಹಾಕಿ."
        ),
        "severity": "severe",
    },

    "Tomato_healthy": {
        "display": "Tomato — Healthy",
        "icon": "✅",
        "treatment_en": (
            "No disease detected. Continue regular monitoring and "
            "maintain good field hygiene."
        ),
        "treatment_hi": (
            "कोई रोग नहीं पाया गया। नियमित निगरानी और अच्छी खेत "
            "स्वच्छता जारी रखें।"
        ),
        "treatment_mr": (
            "कोणताही रोग आढळला नाही. नियमित देखरेख आणि चांगली "
            "शेत स्वच्छता सुरू ठेवा."
        ),
        "treatment_kn": (
            "ಯಾವುದೇ ರೋಗ ಪತ್ತೆಯಾಗಿಲ್ಲ. ನಿಯಮಿತ ಮೇಲ್ವಿಚಾರಣೆ ಮತ್ತು "
            "ಉತ್ತಮ ಹೊಲದ ನೈರ್ಮಲ್ಯವನ್ನು ಮುಂದುವರಿಸಿ."
        ),
        "severity": "healthy",
    },
}


SEVERITY_COLORS = {
    "healthy": "#7bd389",
    "moderate": "#f2c744",
    "severe": "#e0665a",
}

SEVERITY_LABELS = {
    "healthy": "Healthy",
    "moderate": "Moderate risk",
    "severe": "Severe — act now",
}


# ---------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------

@st.cache_resource
def load_my_model():
    """
    Load the TensorFlow/Keras model and class names.
    Cached so the model is not loaded on every interaction.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}"
        )

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Class names file not found: {CLASS_NAMES_PATH.name}"
        )

    try:
        from tensorflow.keras.models import load_model

        model = load_model(MODEL_PATH)

        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
            class_names = json.load(file)

        # Handle either:
        # ["class1", "class2"]
        # OR
        # {"0": "class1", "1": "class2"}
        if isinstance(class_names, dict):
            try:
                class_names = [
                    class_names[str(i)]
                    for i in range(len(class_names))
                ]
            except KeyError:
                class_names = list(class_names.values())

        return model, class_names

    except Exception as error:
        raise RuntimeError(
            f"Could not load the AI model: {error}"
        ) from error


# ---------------------------------------------------------
# HERO SECTION
# ---------------------------------------------------------

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            🌾 KrishiRakshak AI
        </div>

        <div class="hero-sub">
            Early Detection & Management of Crop Diseases
            and Pest Infestations
        </div>

        <div class="hero-badge">
            🏛️ Government of Maharashtra · SIH 2026 · SIH26131
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# LOAD MODEL WITH ERROR HANDLING
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# STATISTICS
# ---------------------------------------------------------

s1, s2, s3, s4 = st.columns(4)

with s1:
    st.markdown(
        """
        <div class="stat-card">
            <div class="stat-num">4</div>
            <div class="stat-label">Disease classes</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with s2:
    st.markdown(
        """
        <div class="stat-card">
            <div class="stat-num">99%+</div>
            <div class="stat-label">Target accuracy</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with s3:
    st.markdown(
        """
        <div class="stat-card">
            <div class="stat-num">4</div>
            <div class="stat-label">Languages</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with s4:
    st.markdown(
        """
        <div class="stat-card">
            <div class="stat-num">AI</div>
            <div class="stat-label">Instant detection</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.write("")


# ---------------------------------------------------------
# HOW IT WORKS
# ---------------------------------------------------------

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
            <b>📷 Snap a photo</b>
            <br>
            <span style="opacity:0.85; font-size:0.85rem;">
                Take a clear photo of the affected leaf
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
            <b>🧠 AI analyzes</b>
            <br>
            <span style="opacity:0.85; font-size:0.85rem;">
                AI model analyzes the crop image
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
            <b>🗣️ Get advice</b>
            <br>
            <span style="opacity:0.85; font-size:0.85rem;">
                Receive treatment guidance in your language
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.write("")


# ---------------------------------------------------------
# IMAGE UPLOAD
# ---------------------------------------------------------

st.markdown(
    '<div class="section-header">📸 Scan a crop leaf</div>',
    unsafe_allow_html=True,
)

col_upload, col_result = st.columns(
    [1, 1.2],
    gap="large",
)


# ---------------------------------------------------------
# LEFT: UPLOAD
# ---------------------------------------------------------

with col_upload:

    st.markdown(
        """
        <div class="upload-panel-label">
            Drag a leaf photo below, or click to browse
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload crop leaf image",
        type=["jpg", "jpeg", "png"],
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
                "The uploaded file could not be read as an image."
            )


# ---------------------------------------------------------
# RIGHT: RESULT
# ---------------------------------------------------------

with col_result:

    if img is not None:

        try:

            # ---------------------------------------------
            # IMAGE PREPROCESSING
            # ---------------------------------------------

            img_resized = img.resize(
                (224, 224),
                Image.Resampling.LANCZOS,
            )

            image_array = np.asarray(
                img_resized,
                dtype=np.float32,
            )

            image_array = image_array / 255.0

            image_array = np.expand_dims(
                image_array,
                axis=0,
            )

            # ---------------------------------------------
            # MODEL PREDICTION
            # ---------------------------------------------

            prediction = model.predict(
                image_array,
                verbose=0,
            )

            prediction = np.asarray(prediction)

            # Handle a normal classification output
            if prediction.ndim == 2:
                prediction = prediction[0]

            predicted_index = int(
                np.argmax(prediction)
            )

            confidence = float(
                np.max(prediction)
            ) * 100

            # Prevent index errors
            if predicted_index >= len(class_names):
                raise ValueError(
                    "Model output does not match class_names.json."
                )

            result = class_names[predicted_index]

            # ---------------------------------------------
            # DISEASE INFORMATION
            # ---------------------------------------------

            info = DISEASE_INFO.get(
                result,
                {
                    "display": str(result).replace("_", " "),
                    "icon": "🌿",

                    "treatment_en": (
                        "The model detected this condition. "
                        "P
