import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image


# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="KrishiRakshak AI",
    page_icon="🌾",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "crop_model.h5"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"


# =========================================================
# CSS
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #14210f, #26391d);
    }

    .hero {
        background: linear-gradient(135deg, #23331b, #4a6339);
        padding: 35px;
        border-radius: 24px;
        margin-bottom: 25px;
        border: 1px solid rgba(255,255,255,.1);
    }

    .hero-title {
        font-size: 42px;
        font-weight: 800;
        color: #f6f9f2;
    }

    .hero-sub {
        font-size: 18px;
        color: #d3ddc7;
        margin-top: 5px;
    }

    .badge {
        display: inline-block;
        margin-top: 15px;
        padding: 7px 16px;
        border-radius: 25px;
        background: #f2c744;
        color: #23331b;
        font-weight: 700;
    }

    .card {
        background: rgba(255,255,255,.07);
        border: 1px solid rgba(255,255,255,.12);
        border-radius: 18px;
        padding: 22px;
        height: 100%;
        box-shadow: 0 5px 18px rgba(0,0,0,.2);
    }

    .stat-number {
        color: #f2c744;
        font-size: 30px;
        font-weight: 800;
    }

    .stat-label {
        color: #d3ddc7;
    }

    .section-title {
        color: #f6f9f2;
        font-size: 25px;
        font-weight: 800;
        margin-top: 30px;
        margin-bottom: 15px;
    }

    .result {
        background: rgba(255,255,255,.07);
        border-radius: 20px;
        padding: 25px;
        border: 1px solid rgba(255,255,255,.12);
    }

    .result-name {
        color: #f6f9f2;
        font-size: 28px;
        font-weight: 800;
    }

    .treatment {
        background: rgba(242,199,68,.10);
        border: 1px solid rgba(242,199,68,.35);
        border-radius: 18px;
        padding: 20px;
        margin-top: 15px;
    }

    .footer {
        text-align: center;
        color: #9fab8f;
        margin-top: 40px;
        padding: 20px;
        border-top: 1px solid rgba(255,255,255,.1);
    }

    .stButton button {
        background: #f2c744 !important;
        color: #23331b !important;
        border-radius: 25px !important;
        font-weight: 700 !important;
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
        "name": "Bell Pepper — Bacterial Spot",
        "icon": "🫑",
        "severity": "Moderate Risk",
        "color": "#f2c744",
        "en": (
            "Use an appropriate copper-based bactericide according "
            "to the product label. Avoid overhead watering and "
            "remove severely infected leaves."
        ),
        "hi": (
            "उत्पाद के लेबल के अनुसार उपयुक्त कॉपर आधारित "
            "बैक्टीरिसाइड का प्रयोग करें। ऊपर से पानी देने से बचें "
            "और अधिक संक्रमित पत्तियों को हटा दें।"
        ),
        "mr": (
            "उत्पादनाच्या लेबलनुसार योग्य तांबेयुक्त बॅक्टेरिसाइड "
            "वापरा. वरून पाणी देणे टाळा आणि संक्रमित पाने काढून टाका."
        ),
        "kn": (
            "ಉತ್ಪನ್ನದ ಲೇಬಲ್ ಪ್ರಕಾರ ಸೂಕ್ತವಾದ ತಾಮ್ರ ಆಧಾರಿತ "
            "ಬ್ಯಾಕ್ಟೀರಿಸೈಡ್ ಬಳಸಿ. ಮೇಲಿನಿಂದ ನೀರುಣಿಸುವುದನ್ನು ತಪ್ಪಿಸಿ."
        ),
    },

    "Potato___Early_blight": {
        "name": "Potato — Early Blight",
        "icon": "🥔",
        "severity": "Moderate Risk",
        "color": "#f2c744",
        "en": (
            "Use an appropriate fungicide according to the product "
            "label. Remove infected plant debris and maintain good "
            "field hygiene."
        ),
        "hi": (
            "उत्पाद के लेबल के अनुसार उपयुक्त फफूंदनाशक का प्रयोग "
            "करें। संक्रमित पौधों के अवशेष हटाएं।"
        ),
        "mr": (
            "उत्पादनाच्या लेबलनुसार योग्य बुरशीनाशक वापरा. "
            "संक्रमित अवशेष काढून टाका."
        ),
        "kn": (
            "ಉತ್ಪನ್ನದ ಲೇಬಲ್ ಪ್ರಕಾರ ಸೂಕ್ತವಾದ ಶಿಲೀಂಧ್ರನಾಶಕ ಬಳಸಿ. "
            "ಸೋಂಕಿತ ಸಸ್ಯದ ಅವಶೇಷಗಳನ್ನು ತೆಗೆದುಹಾಕಿ."
        ),
    },

    "Tomato_Late_blight": {
        "name": "Tomato — Late Blight",
        "icon": "🍅",
        "severity": "Severe — Act Now",
        "color": "#e0665a",
        "en": (
            "Use an appropriate fungicide according to the product "
            "label and remove severely infected plant material."
        ),
        "hi": (
            "उत्पाद के लेबल के अनुसार उपयुक्त फफूंदनाशक का प्रयोग "
            "करें और संक्रमित पौधों के हिस्सों को हटा दें।"
        ),
        "mr": (
            "उत्पादनाच्या लेबलनुसार योग्य बुरशीनाशक वापरा आणि "
            "संक्रमित भाग काढून टाका."
        ),
        "kn": (
            "ಉತ್ಪನ್ನದ ಲೇಬಲ್ ಪ್ರಕಾರ ಸೂಕ್ತವಾದ ಶಿಲೀಂಧ್ರನಾಶಕ ಬಳಸಿ "
            "ಮತ್ತು ಸೋಂಕಿತ ಭಾಗಗಳನ್ನು ತೆಗೆದುಹಾಕಿ."
        ),
    },

    "Tomato_healthy": {
        "name": "Tomato — Healthy",
        "icon": "✅",
        "severity": "Healthy",
        "color": "#7bd389",
        "en": (
            "No disease detected. Continue regular monitoring "
            "and maintain good field hygiene."
        ),
        "hi": (
            "कोई रोग नहीं पाया गया। नियमित निगरानी और अच्छी "
            "खेत स्वच्छता जारी रखें।"
        ),
        "mr": (
            "कोणताही रोग आढळला नाही. नियमित देखरेख आणि "
            "चांगली शेत स्वच्छता सुरू ठेवा."
        ),
        "kn": (
            "ಯಾವುದೇ ರೋಗ ಪತ್ತೆಯಾಗಿಲ್ಲ. ನಿಯಮಿತ ಮೇಲ್ವಿಚಾರಣೆ "
            "ಮತ್ತು ಉತ್ತಮ ಹೊಲದ ನೈರ್ಮಲ್ಯವನ್ನು ಮುಂದುವರಿಸಿ."
        ),
    },
}


# =========================================================
# MODEL
# =========================================================

@st.cache_resource
def load_model_and_classes():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "crop_model.h5 was not found."
        )

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            "class_names.json was not found."
        )

    from tensorflow.keras.models import load_model

    model = load_model(MODEL_PATH)

    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        classes = json.load(file)

    if isinstance(classes, dict):
        try:
            classes = [
                classes[str(i)]
                for i in range(len(classes))
            ]
        except KeyError:
            classes = list(classes.values())

    return model, classes


# =========================================================
# HERO
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🌾 KrishiRakshak AI</div>

        <div class="hero-sub">
            Early Detection & Management of Crop Diseases
            and Pest Infestations
        </div>

        <div class="badge">
            🏛️ Government of Maharashtra · SIH 2026 · SIH26131
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD AI MODEL
# =========================================================

try:
    model, class_names = load_model_and_classes()

except Exception as error:
    st.error("⚠️ AI model could not be loaded.")
    st.code(str(error))

    st.info(
        "Make sure crop_model.h5 and class_names.json "
        "are in the same folder as app.py."
    )

    st.stop()


# =========================================================
# STATISTICS
# =========================================================

st1, st2, st3, st4 = st.columns(4)

statistics = [
    ("4", "Disease Classes"),
    ("AI", "Image Detection"),
    ("4", "Languages"),
    ("224×224", "Input Size"),
]

for column, (number, label) in zip(
    [st1, st2, st3, st4],
    statistics,
):

    with column:
        st.markdown(
            f"""
            <div class="card">
                <div class="stat-number">{number}</div>
                <div class="stat-label">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# HOW IT WORKS
# =========================================================

st.markdown(
    '<div class="section-title">⚡ How it works</div>',
    unsafe_allow_html=True,
)

step1, step2, step3 = st.columns(3)

steps = [
    (
        "1",
        "📷 Snap a photo",
        "Take a clear photo of the affected crop leaf.",
    ),
    (
        "2",
        "🧠 AI analyzes",
        "The trained AI model analyzes the image.",
    ),
    (
        "3",
        "🗣️ Get advice",
        "Receive disease information and guidance.",
    ),
]

for column, (number, title, description) in zip(
    [step1, step2, step3],
    steps,
):

    with column:
        st.markdown(
            f"""
            <div class="card">
                <h3>{number}. {title}</h3>
                <p>{description}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# UPLOAD SECTION
# =========================================================

st.markdown(
    '<div class="section-title">📸 Scan a crop leaf</div>',
    unsafe_allow_html=True,
)

upload_column, result_column = st.columns(
    [1, 1.2],
    gap="large",
)


# =========================================================
# UPLOAD
# =========================================================

with upload_column:

    st.write(
        "Upload a clear JPG, JPEG, or PNG image."
    )

    uploaded_file = st.file_uploader(
        "Choose a crop leaf image",
        type=["jpg", "jpeg", "png"],
    )

    image = None

    if uploaded_file is not None:

        try:

            image = Image.open(
                uploaded_file
            ).convert("RGB")

            st.image(
                image,
                caption="Uploaded crop image",
                use_container_width=True,
            )

        except Exception as error:

            st.error(
                "Unable to read this image."
            )

            st.code(str(error))


# =========================================================
# RESULT
# =========================================================

with result_column:

    if image is None:

        st.markdown(
            """
            <div class="result">
                <h2>🌱 Ready to analyze</h2>
                <p>
                    Upload a crop leaf image and the AI
                    prediction will appear here.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        try:

            # ---------------------------------------------
            # PREPROCESSING
            # ---------------------------------------------

            resized_image = image.resize(
                (224, 224),
                Image.Resampling.LANCZOS,
            )

            image_array = np.asarray(
                resized_image,
                dtype=np.float32,
            )

            image_array /= 255.0

            image_array = np.expand_dims(
                image_array,
                axis=0,
            )

            # ---------------------------------------------
            # PREDICTION
            # ---------------------------------------------

            prediction = model.predict(
                image_array,
                verbose=0,
            )

            prediction = np.asarray(
                prediction
            )

            if prediction.ndim == 2:
                prediction = prediction[0]

            predicted_index = int(
                np.argmax(prediction)
            )

            confidence = float(
                np.max(prediction)
            ) * 100

            if predicted_index >= len(class_names):
                raise ValueError(
                    "Model output count does not match "
                    "class_names.json."
                )

            predicted_class = str(
                class_names[predicted_index]
            )

            # ---------------------------------------------
            # DISEASE DETAILS
            # ---------------------------------------------

            info = DISEASE_INFO.get(
                predicted_class
            )

            if info is None:

                info = {
                    "name": predicted_class.replace(
                        "_",
                        " ",
                    ),
                    "icon": "🌿",
                    "severity": "Unknown",
                    "color": "#f2c744",
                    "en": (
                        "The AI detected this condition. "
                        "Please consult a local agricultural "
                        "expert for confirmation."
                    ),
                    "hi": (
                        "AI ने इस स्थिति का पता लगाया है। "
                        "पुष्टि के लिए स्थानीय कृषि विशेषज्ञ "
                        "से सलाह लें।"
                    ),
                    "mr": (
                        "AI ने ही स्थिती ओळखली आहे. "
                        "पुष्टीसाठी स्थानिक कृषी तज्ञांचा "
                        "सल्ला घ्या."
                    ),
                    "kn": (
                        "AI ಈ ಸ್ಥಿತಿಯನ್ನು ಪತ್ತೆಹಚ್ಚಿದೆ. "
                        "ದೃಢೀಕರಣಕ್ಕಾಗಿ ಸ್ಥಳೀಯ ಕೃಷಿ ತಜ್ಞರನ್ನು "
                        "ಸಂಪರ್ಕಿಸಿ."
                    ),
                }

            confidence = max(
                0,
                min(
                    confidence,
                    100,
                ),
            )

            # ---------------------------------------------
            # RESULT CARD
            # ---------------------------------------------

            st.markdown(
                f"""
                <div class="result"
                     style="border-top:5px solid {info['color']};">

                    <div style="
                        color:#bcc7ab;
                        font-size:13px;
                        text-transform:uppercase;
                        letter-spacing:1px;
                    ">
                        Detection Result
                    </div>

                    <div class="result-name">
                        {info['icon']} {info['name']}
                    </div>

                    <div style="
                        color:{info['color']};
                        font-weight:700;
                        margin-top:10px;
                    ">
                        ● {info['severity']}
                    </div>

                    <p>
                        Confidence:
                        <b>{confidence:.1f}%</b>
                    </p>

                    <div style="
                        background:rgba(255,255,255,.12);
                        height:15px;
                        border-radius:10px;
                        overflow:hidden;
                    ">
                        <div style="
                            width:{confidence:.1f}%;
                            height:15px;
                            background:{info['color']};
                        "></div>
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

            # ---------------------------------------------
            # LANGUAGE
            # ---------------------------------------------

            language = st.selectbox(
                "🌐 Advice language",
                [
                    "English",
                    "हिंदी (Hindi)",
                    "मराठी (Marathi)",
                    "ಕನ್ನಡ (Kannada)",
                ],
            )

            language_key = {
                "English": "en",
                "हिंदी (Hindi)": "hi",
                "मराठी (Marathi)": "mr",
                "ಕನ್ನಡ (Kannada)": "kn",
            }

            advice = info[
                language_key[language]
            ]

            # ---------------------------------------------
            # TREATMENT
            # ---------------------------------------------

            st.markdown(
                f"""
                <div class="treatment">
                    <b style="color:#f2c744;">
                        💊 Recommended Action
                    </b>

                    <p>{advice}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ---------------------------------------------
            # VOICE
            # ---------------------------------------------

            if st.button(
                "🔊 Play voice advice",
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

                    tts = gTTS(
                        text=(
                            info["name"]
                            + ". "
                            + advice
                        ),
                        lang=language_codes[
                            language
                        ],
                    )

                    audio_file = (
                        BASE_DIR /
                        "voice_advice.mp3"
                    )

                    tts.save(
                        str(audio_file)
                    )

                    with open(
                        audio_file,
                        "rb",
                    ) as audio:

                        st.audio(
                            audio.read(),
                            format="audio/mp3",
                        )

                except Exception as error:

                    st.warning(
                        "Voice generation is unavailable. "
                        "Check your internet connection."
                    )

                    st.caption(
                        str(error)
                    )

        except Exception as error:

            st.error(
                "⚠️ Image analysis failed."
            )

            st.code(str(error))


# =========================================================
# HELPLINE
# =========================================================

st.markdown(
    '<div class="section-title">📞 Farmer Helpline</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="card">

        <h3>📱 Kisan Call Centre</h3>

        <p>
            Toll-free:
            <b>1800-180-1551</b>
        </p>

        <p>
            Available 6 AM – 10 PM,
            all 7 days.
        </p>

        <hr>

        <h3>📱 PM-KISAN Helpline</h3>

        <p>
            <b>155261</b> /
            <b>1800-115-526</b>
        </p>

        <p>
            Phone:
            <b>011-24300606</b>
        </p>

    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# FUTURE CROPS
# =========================================================

st.markdown(
    '<div class="section-title">🌱 Future Crop Coverage</div>',
    unsafe_allow_html=True,
)

crop1, crop2, crop3, crop4 = st.columns(4)

future_crops = [
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
    (
        "🎋",
        "Sugarcane",
        "Red rot, smut",
    ),
]

for column, crop_data in zip(
    [crop1, crop2, crop3, crop4],
    future_crops,
):

    icon, crop_name, diseases = crop_data

    with column:

        st.markdown(
            f"""
            <div class="card">

                <h3>
                    {icon} {crop_name}
                </h3>

                <p>
                    {diseases}
                </p>

            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        Prototype for SIH 2026 ·
        Problem Statement SIH26131 ·
        Government of Maharashtra
    </div>
    """,
    unsafe_allow_html=True,
)
