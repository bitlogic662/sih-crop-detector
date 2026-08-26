import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="KrishiRakshak AI",
    page_icon="🌾",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "crop_model.h5"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"

st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top left, #2b3d22 0%, #1c2a17 55%, #141f10 100%);
}
.stApp p, .stApp span, .stApp label, .stApp li, .stMarkdown, .stCaption {
    color: #eef2e6 !important;
}
.hero {
    background: linear-gradient(135deg, #23331b 0%, #35492a 45%, #4a6339 100%);
    padding: 2.2rem 2.5rem;
    border-radius: 24px;
    margin-bottom: 1.5rem;
    box-shadow: 0 12px 32px rgba(0,0,0,.35);
    border: 1px solid rgba(255,255,255,.08);
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    color: #f6f9f2;
}
.hero-sub {
    font-size: 1.05rem;
    color: #d3ddc7;
}
.badge {
    display: inline-block;
    background: #f2c744;
    color: #23331b;
    font-weight: 700;
    padding: 6px 16px;
    border-radius: 30px;
    font-size: .8rem;
    margin-top: 12px;
}
.stat-card,
.step-card,
.roadmap-card,
.helpline-card,
.result-card {
    background: rgba(255,255,255,.06);
    border: 1px solid rgba(255,255,255,.10);
    border-radius: 18px;
    box-shadow: 0 5px 18px rgba(0,0,0,.22);
}
.stat-card {
    padding: 1rem;
    text-align: center;
}
.stat-num {
    font-size: 1.8rem;
    font-weight: 800;
    color: #f2c744;
}
.stat-label {
    font-size: .82rem;
    color: #d3ddc7;
}
.section {
    font-size: 1.35rem;
    font-weight: 800;
    margin: 1.8rem 0 .9rem;
    color: #f6f9f2;
}
.step-card {
    padding: 1.2rem;
    text-align: center;
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
    margin: 0 auto 10px;
}
.result-card {
    padding: 1.5rem;
    border-top: 5px solid #f2c744;
}
.result-label {
    font-size: .78rem;
    letter-spacing: 1px;
    color: #bcc7ab;
    text-transform: uppercase;
}
.result-name {
    font-size: 1.65rem;
    font-weight: 800;
    margin-top: 5px;
}
.conf-bg {
    background: rgba(255,255,255,.12);
    border-radius: 10px;
    height: 15px;
    margin-top: 8px;
    overflow: hidden;
}
.conf-fill {
    height: 15px;
    border-radius: 10px;
}
.treatment {
    background: rgba(242,199,68,.10);
    border: 1px solid rgba(242,199,68,.35);
    border-radius: 18px;
    padding: 1.2rem 1.4rem;
    margin-top: 1rem;
}
.helpline-card {
    padding: 1.3rem 1.5rem;
}
.roadmap-card {
    padding: 1rem;
    height: 100%;
}
.stButton button {
    background: #f2c744 !important;
    color: #23331b !important;
    border: none !important;
    border-radius: 30px !important;
    font-weight: 700 !important;
}
.footer {
    text-align: center;
    color: #9fab8f;
    margin-top: 2.5rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(255,255,255,.1);
}
</style>
""", unsafe_allow_html=True)


DISEASE_INFO = {
    "Pepper__bell___Bacterial_spot": {
        "display": "Bell Pepper — Bacterial Spot",
        "icon": "🫑",
        "severity": "moderate",
        "en": "Use an appropriate copper-based bactericide according to the product label. Avoid overhead watering and remove severely infected leaves.",
        "hi": "उत्पाद के लेबल के अनुसार उपयुक्त कॉपर आधारित बैक्टीरिसाइड का प्रयोग करें। ऊपर से पानी देने से बचें और अधिक संक्रमित पत्तियों को हटा दें।",
        "mr": "उत्पादनाच्या लेबलनुसार योग्य तांबेयुक्त बॅक्टेरिसाइड वापरा. वरून पाणी देणे टाळा आणि जास्त संक्रमित पाने काढून टाका.",
        "kn": "ಉತ್ಪನ್ನದ ಲೇಬಲ್ ಪ್ರಕಾರ ಸೂಕ್ತವಾದ ತಾಮ್ರ ಆಧಾರಿತ ಬ್ಯಾಕ್ಟೀರಿಸೈಡ್ ಬಳಸಿ. ಮೇಲಿನಿಂದ ನೀರುಣಿಸುವುದನ್ನು ತಪ್ಪಿಸಿ ಮತ್ತು ಹೆಚ್ಚು ಸೋಂಕಿತ ಎಲೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ.",
    },

    "Potato___Early_blight": {
        "display": "Potato — Early Blight",
        "icon": "🥔",
        "severity": "moderate",
        "en": "Use an appropriate fungicide according to the product label. Remove infected plant debris and maintain good field hygiene.",
        "hi": "उत्पाद के लेबल के अनुसार उपयुक्त फफूंदनाशक का प्रयोग करें। संक्रमित पौधों के अवशेष हटाएं और खेत की स्वच्छता बनाए रखें।",
        "mr": "उत्पादनाच्या लेबलनुसार योग्य बुरशीनाशक वापरा. संक्रमित वनस्पतींचे अवशेष काढून टाका आणि शेताची स्वच्छता राखा.",
        "kn": "ಉತ್ಪನ್ನದ ಲೇಬಲ್ ಪ್ರಕಾರ ಸೂಕ್ತವಾದ ಶಿಲೀಂಧ್ರನಾಶಕ ಬಳಸಿ. ಸೋಂಕಿತ ಸಸ್ಯದ ಅವಶೇಷಗಳನ್ನು ತೆಗೆದುಹಾಕಿ ಮತ್ತು ಹೊಲದ ಸ್ವಚ್ಛತೆಯನ್ನು ಕಾಪಾಡಿ.",
    },

    "Tomato_Late_blight": {
        "display": "Tomato — Late Blight",
        "icon": "🍅",
        "severity": "severe",
        "en": "Use an appropriate fungicide according to the product label and remove severely infected plant material to reduce disease spread.",
        "hi": "उत्पाद के लेबल के अनुसार उपयुक्त फफूंदनाशक का प्रयोग करें और रोग के फैलाव को कम करने के लिए अधिक संक्रमित पौधों के हिस्सों को हटा दें।",
        "mr": "उत्पादनाच्या लेबलनुसार योग्य बुरशीनाशक वापरा आणि रोगाचा प्रसार कमी करण्यासाठी जास्त संक्रमित भाग काढून टाका.",
        "kn": "ಉತ್ಪನ್ನದ ಲೇಬಲ್ ಪ್ರಕಾರ ಸೂಕ್ತವಾದ ಶಿಲೀಂಧ್ರನಾಶಕ ಬಳಸಿ ಮತ್ತು ರೋಗ ಹರಡುವಿಕೆಯನ್ನು ಕಡಿಮೆ ಮಾಡಲು ಹೆಚ್ಚು ಸೋಂಕಿತ ಭಾಗಗಳನ್ನು ತೆಗೆದುಹಾಕಿ.",
    },

    "Tomato_healthy": {
        "display": "Tomato — Healthy",
        "icon": "✅",
        "severity": "healthy",
        "en": "No disease detected. Continue regular monitoring and maintain good field hygiene.",
        "hi": "कोई रोग नहीं पाया गया। नियमित निगरानी और अच्छी खेत स्वच्छता जारी रखें।",
        "mr": "कोणताही रोग आढळला नाही. नियमित देखरेख आणि चांगली शेत स्वच्छता सुरू ठेवा.",
        "kn": "ಯಾವುದೇ ರೋಗ ಪತ್ತೆಯಾಗಿಲ್ಲ. ನಿಯಮಿತ ಮೇಲ್ವಿಚಾರಣೆ ಮತ್ತು ಉತ್ತಮ ಹೊಲದ ನೈರ್ಮಲ್ಯವನ್ನು ಮುಂದುವರಿಸಿ.",
    },
}


COLORS = {
    "healthy": "#7bd389",
    "moderate": "#f2c744",
    "severe": "#e0665a",
}

LABELS = {
    "healthy": "Healthy",
    "moderate": "Moderate risk",
    "severe": "Severe — act now",
}


@st.cache_resource
def load_my_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing {MODEL_PATH.name}"
        )

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CLASS_NAMES_PATH.name}"
        )

    from tensorflow.keras.models import load_model

    model = load_model(MODEL_PATH)

    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        class_names = json.load(f)

    if isinstance(class_names, dict):

        try:
            class_names = [
                class_names[str(i)]
                for i in range(len(class_names))
            ]

        except KeyError:
            class_names = list(
                class_names.values()
            )

    return model, class_names


# ---------------------------------------------------------
# HERO
# ---------------------------------------------------------

st.markdown("""
<div class="hero">

    <div class="hero-title">
        🌾 KrishiRakshak AI
    </div>

    <div class="hero-sub">
        Early Detection & Management of Crop Diseases
        and Pest Infestations
    </div>

    <div class="badge">
        🏛️ Government of Maharashtra · SIH 2026 · SIH26131
    </div>

</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

try:

    model, class_names = load_my_model()

except Exception as e:

    st.error("⚠️ AI model could not be loaded.")

    st.code(str(e))

    st.info(
        "Keep crop_model.h5 and class_names.json "
        "in the same folder as app.py."
    )

    st.stop()


# ---------------------------------------------------------
# STATISTICS
# ---------------------------------------------------------

s1, s2, s3, s4 = st.columns(4)

stats = [
    ("4", "Disease classes"),
    ("AI", "Image detection"),
    ("4", "Languages"),
    ("224×224", "Input image"),
]

for col, (num, label) in zip(
    (s1, s2, s3, s4),
    stats
):

    with col:

        st.markdown(
            f"""
            <div class="stat-card">

                <div class="stat-num">
                    {num}
                </div>

                <div class="stat-label">
                    {label}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------
# HOW IT WORKS
# ---------------------------------------------------------

st.markdown(
    '<div class="section">⚡ How it works</div>',
    unsafe_allow_html=True,
)

h1, h2, h3 = st.columns(3)

steps = [
    (
        "1",
        "📷 Snap a photo",
        "Take a clear photo of the affected leaf",
    ),
    (
        "2",
        "🧠 AI analyzes",
        "The trained model analyzes the image",
    ),
    (
        "3",
        "🗣️ Get advice",
        "Receive guidance in your language",
    ),
]

for col, (num, title, text) in zip(
    (h1, h2, h3),
    steps
):

    with col:

        st.markdown(
            f"""
            <div class="step-card">

                <div class="step-num">
                    {num}
                </div>

                <b>{title}</b>

                <br>

                <span>
                    {text}
                </span>

            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------
# IMAGE UPLOAD
# ---------------------------------------------------------

st.markdown(
    '<div class="section">📸 Scan a crop leaf</div>',
    unsafe_allow_html=True,
)

upload_col, result_col = st.columns(
    [1, 1.2],
    gap="large",
)


# ---------------------------------------------------------
# UPLOAD COLUMN
# ---------------------------------------------------------

with upload_col:

    st.write(
        "Upload a JPG, JPEG, or PNG image of a crop leaf."
    )

    uploaded = st.file_uploader(
        "Choose a crop leaf image",
        type=[
            "jpg",
            "jpeg",
            "png",
        ],
    )

    image = None

    if uploaded is not None:

        try:

            image = Image.open(
                uploaded
            ).convert("RGB")

            st.image(
                image,
                caption="Uploaded crop image",
                use_container_width=True,
            )

        except Exception as e:

            st.error(
                "Could not read the uploaded image."
            )

            st.code(str(e))


# ---------------------------------------------------------
# RESULT COLUMN
# ---------------------------------------------------------

with result_col:

    if image is None:

        st.markdown(
            """
            <div class="result-card"
                 style="
                 min-height:300px;
                 display:flex;
                 align-items:center;
                 justify-content:center;
                 text-align:center;
                 ">

                <div>

                    <div style="font-size:3rem;">
                        🌱
                    </div>

                    <div>
                        Upload a crop image to see
                        the AI result.
                    </div>

                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        try:

            # -----------------------------
            # PREPROCESS IMAGE
            # -----------------------------

            resized = image.resize(
                (224, 224),
                Image.Resampling.LANCZOS,
            )

            array = np.asarray(
                resized,
                dtype=np.float32,
            )

            array = array / 255.0

            array = np.expand_dims(
                array,
                axis=0,
            )

            # -----------------------------
            # PREDICTION
            # -----------------------------

            prediction = np.asarray(
                model.predict(
                    array,
                    verbose=0,
                )
            )

            if prediction.ndim == 2:
                prediction = prediction[0]

            index = int(
                np.argmax(prediction)
            )

            confidence = (
                float(
                    np.max(prediction)
                ) * 100
            )

            if index >= len(class_names):

                raise ValueError(
                    "The number of model outputs "
                    "does not match class_names.json."
                )

            result = str(
                class_names[index]
            )

            # -----------------------------
            # GET DISEASE INFORMATION
            # -----------------------------

            info = DISEASE_INFO.get(
                result,
                {
                    "display": result.replace(
                        "_",
                        " ",
                    ),
                    "icon": "🌿",
                    "severity": "moderate",

                    "en": (
                        "Please consult a local "
                        "agriculture expert before "
                        "taking treatment action."
                    ),

                    "hi": (
                        "उपचार करने से पहले स्थानीय "
                        "कृषि विशेषज्ञ से सलाह लें।"
                    ),

                    "mr": (
                        "उपचार करण्यापूर्वी स्थानिक "
                        "कृषी तज्ञांचा सल्ला घ्या."
                    ),

                    "kn": (
                        "ಚಿಕಿತ್ಸೆ ಕೈಗೊಳ್ಳುವ ಮೊದಲು "
                        "ಸ್ಥಳೀಯ ಕೃಷಿ ತಜ್ಞರನ್ನು ಸಂಪರ್ಕಿಸಿ."
                    ),
                },
            )

            severity = info["severity"]

            color = COLORS[severity]

            label = LABELS[severity]

            bar_width = min(
                max(
                    confidence,
                    0,
                ),
                100,
            )

            # -----------------------------
            # RESULT CARD
            # -----------------------------

            st.markdown(
                f"""
                <div class="result-card"
                     style="
                     border-top-color:{color};
                     ">

                    <div class="result-label">
                        Detection result
                    </div>

                    <div class="result-name">
                        {info["icon"]}
                        {info["display"]}
                    </div>

                    <div
                        style="
                        margin-top:10px;
                        color:{color};
                        font-weight:700;
                        ">

                        ● {label}

                    </div>

                    <div
                        style="
                        margin-top:14px;
                        ">

                        Confidence:
                        <b>{confidence:.1f}%</b>

                    </div>

                    <div class="conf-bg">

                        <div
                            class="conf-fill"
                            style="
                            width:{bar_width:.1f}%;
                            background:{color};
                            ">
                        </div>

                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


            # -----------------------------
            # LANGUAGE
            # -----------------------------

            language = st.selectbox(
                "🌐 Advice language",
                [
                    "English",
                    "हिंदी (Hindi)",
                    "मराठी (Marathi)",
                    "ಕನ್ನಡ (Kannada)",
                ],
            )

            key_map = {
                "English": "en",
                "हिंदी (Hindi)": "hi",
                "मराठी (Marathi)": "mr",
                "ಕನ್ನಡ (Kannada)": "kn",
            }

            advice = info[
                key_map[language]
            ]


            # -----------------------------
            # TREATMENT
            # -----------------------------

            st.markdown(
                f"""
                <div class="treatment">

                    <b style="color:#f2c744;">
                        💊 Recommended action
                    </b>

                    <br><br>

                    {advice}

                </div>
                """,
                unsafe_allow_html=True,
            )


            # -----------------------------
            # VOICE
            # -----------------------------

            if st.button(
                "🔊 Play voice advice",
                use_container_width=True,
            ):

                try:

                    from gtts import gTTS

                    lang_codes = {
                        "English": "en",
                        "हिंदी (Hindi)": "hi",
                        "मराठी (Marathi)": "mr",
                        "ಕನ್ನಡ (Kannada)": "kn",
                    }

                    audio = gTTS(
                        f"{info['display']}. {advice}",
                        lang=lang_codes[language],
                    )

                    audio_path = (
                        BASE_DIR /
                        "voice_advice.mp3"
                    )

                    audio.save(
                        audio_path
                    )

                    with open(
                        audio_path,
                        "rb",
                    ) as f:

                        st.audio(
                            f.read(),
                            format="audio/mp3",
                        )

                except Exception as e:

                    st.warning(
                        "Voice generation needs an "
                        "internet connection and may "
                        "be unavailable."
                    )

                    st.caption(
                        str(e)
                    )


        except Exception as e:

            st.error(
                "⚠️ Image analysis failed."
            )

            st.code(
                str(e)
            )


# ---------------------------------------------------------
# HELPLINE
# ---------------------------------------------------------

st.markdown(
    '<div class="section">📞 Farmer helpline & support</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="helpline-card">

        <b style="color:#f2c744;">
            📱 Kisan Call Centre
        </b>

        <br>

        Toll-free:
        <b>1800-180-1551</b>

        · 6 AM–10 PM · All 7 days

        <br><br>

        <b style="color:#f2c744;">
            📱 PM-KISAN Helpline
        </b>

        <br>

        <b>155261</b> /
        <b>1800-115-526</b>

        · ☎️ <b>011-24300606</b>

    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "For demonstration purposes. Verify treatment "
    "recommendations with a qualified agricultural "
    "expert before applying pesticides or fungicides."
)


# ---------------------------------------------------------
# FUTURE CROP COVERAGE
# ---------------------------------------------------------

st.markdown(
    '<div class="section">🌱 Future crop coverage</div>',
    unsafe_allow_html=True,
)

r1, r2, r3, r4 = st.columns(4)

future = [
    (
        "🌾",
        "Jowar",
        "Grain mold, downy mildew",
    ),
    (
        "🌾",
        "Rice",
        "Blast, bacterial blight",
    ),
    (
        "🌿",
        "Cotton",
        "Pink bollworm, leaf curl",
    ),
