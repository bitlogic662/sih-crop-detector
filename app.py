import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import json

st.set_page_config(
    page_title="KrishiRakshak AI - Crop Disease Detection",
    page_icon="🌾",
    layout="centered"
)

# ---------- Custom styling ----------
st.markdown("""
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1b4332;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .result-card {
        background-color: #f0f7f0;
        border-left: 6px solid #2d6a4f;
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    .confidence-bar-bg {
        background-color: #e0e0e0;
        border-radius: 8px;
        height: 14px;
        width: 100%;
        margin-top: 6px;
    }
    .confidence-bar-fill {
        background-color: #2d6a4f;
        height: 14px;
        border-radius: 8px;
    }
    .treatment-box {
        background-color: #fff8e6;
        border-left: 6px solid #e0a800;
        padding: 1rem 1.3rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    .footer-note {
        color: #888;
        font-size: 0.8rem;
        margin-top: 3rem;
        text-align: center;
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

# ---------- Header ----------
st.markdown('<p class="main-title">🌾 KrishiRakshak AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Early Detection & Management of Crop Diseases and Pest Infestations — Government of Maharashtra, SIH 2026</p>', unsafe_allow_html=True)

st.markdown("Upload a photo of a crop leaf below. The model analyzes it offline-capable, without needing internet once deployed on-device.")

# ---------- Disease info: name, treatment, translations ----------
disease_info = {
    "Pepper__bell___Bacterial_spot": {
        "display": "Bell Pepper — Bacterial Spot",
        "treatment_en": "Apply copper-based bactericide. Avoid overhead watering and remove infected leaves.",
        "treatment_hi": "कॉपर आधारित बैक्टीरिसाइड का प्रयोग करें। ऊपर से पानी देने से बचें और संक्रमित पत्तियों को हटा दें।",
        "treatment_mr": "तांबेयुक्त बॅक्टेरिसाइड वापरा. वरून पाणी देणे टाळा आणि संक्रमित पाने काढून टाका.",
        "severity": "moderate"
    },
    "Potato___Early_blight": {
        "display": "Potato — Early Blight",
        "treatment_en": "Apply fungicide (Chlorothalonil or Mancozeb). Rotate crops and remove infected debris.",
        "treatment_hi": "फफूंदनाशक (क्लोरोथालोनिल या मैंकोजेब) का प्रयोग करें। फसल चक्र अपनाएं और संक्रमित अवशेष हटा दें।",
        "treatment_mr": "बुरशीनाशक (क्लोरोथॅलोनिल किंवा मॅन्कोझेब) वापरा. पीक फेरपालट करा आणि संक्रमित अवशेष काढून टाका.",
        "severity": "moderate"
    },
    "Tomato_Late_blight": {
        "display": "Tomato — Late Blight",
        "treatment_en": "Apply copper-based fungicide immediately. Remove and destroy infected plants to prevent spread.",
        "treatment_hi": "तुरंत कॉपर आधारित फफूंदनाशक का प्रयोग करें। फैलाव रोकने के लिए संक्रमित पौधों को हटाकर नष्ट कर दें।",
        "treatment_mr": "त्वरित तांबेयुक्त बुरशीनाशक वापरा. प्रसार रोखण्यासाठी संक्रमित रोपे काढून नष्ट करा.",
        "severity": "severe"
    },
    "Tomato_healthy": {
        "display": "Tomato — Healthy",
        "treatment_en": "No disease detected. Continue regular monitoring and good field hygiene.",
        "treatment_hi": "कोई रोग नहीं पाया गया। नियमित निगरानी और अच्छी खेत स्वच्छता जारी रखें।",
        "treatment_mr": "कोणताही रोग आढळला नाही. नियमित देखरेख आणि चांगली शेत स्वच्छता सुरू ठेवा.",
        "severity": "healthy"
    }
}

severity_colors = {
    "healthy": "#2d6a4f",
    "moderate": "#e0a800",
    "severe": "#c0392b"
}

# ---------- Upload ----------
uploaded = st.file_uploader("Upload a leaf/crop photo", type=["jpg", "png", "jpeg"])

if uploaded:
    col1, col2 = st.columns([1, 1])
    with col1:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded image", use_container_width=True)

    img_resized = img.resize((224, 224))
    arr = np.expand_dims(np.array(img_resized) / 255.0, axis=0)
    pred = model.predict(arr)
    result = class_names[np.argmax(pred)]
    confidence = float(np.max(pred)) * 100
    info = disease_info.get(result, {
        "display": result.replace("_", " "),
        "treatment_en": "Consult a local agriculture expert.",
        "treatment_hi": "स्थानीय कृषि विशेषज्ञ से सलाह लें।",
        "treatment_mr": "स्थानिक कृषी तज्ञांचा सल्ला घ्या.",
        "severity": "moderate"
    })

    with col2:
        color = severity_colors.get(info["severity"], "#2d6a4f")
        st.markdown(f"""
            <div class="result-card" style="border-left-color:{color};">
                <div style="font-size:0.85rem; color:#666;">DETECTION RESULT</div>
                <div style="font-size:1.3rem; font-weight:700; color:{color}; margin-top:4px;">
                    {info['display']}
                </div>
                <div style="margin-top:10px; font-size:0.85rem; color:#666;">Confidence: {confidence:.1f}%</div>
                <div class="confidence-bar-bg">
                    <div class="confidence-bar-fill" style="width:{confidence}%; background-color:{color};"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    lang_choice = st.selectbox("🌐 Language / भाषा / भाषा", ["English", "हिंदी (Hindi)", "मराठी (Marathi)"])
    lang_map = {"English": ("en", "treatment_en"), "हिंदी (Hindi)": ("hi", "treatment_hi"), "मराठी (Marathi)": ("mr", "treatment_mr")}
    lang_code, treatment_key = lang_map[lang_choice]
    treatment_text = info[treatment_key]

    st.markdown(f"""
        <div class="treatment-box">
            <div style="font-weight:600; color:#7a5c00; margin-bottom:4px;">💊 Recommended action</div>
            <div style="color:#444;">{treatment_text}</div>
        </div>
    """, unsafe_allow_html=True)

    if st.button("🔊 Play voice advice"):
        from gtts import gTTS
        tts = gTTS(f"{info['display']}. {treatment_text}", lang=lang_code)
        tts.save("output.mp3")
        st.audio("output.mp3")

else:
    st.info("👆 Upload a leaf photo to get started, or try a sample from the field.")

# ---------- Helpline section ----------
st.markdown("---")
st.markdown("### 📞 Farmer helpline & support")
st.markdown("""
    <div style="background-color:#eef6ff; border-left:6px solid #1d6fa5; padding:1rem 1.3rem; border-radius:10px;">
        <div style="font-weight:600; color:#0c4a75; margin-bottom:8px;">Kisan Call Centre (Government of India)</div>
        <div style="color:#333; font-size:0.95rem;">📱 Toll-free: <b>1800-180-1551</b> — available 6 AM to 10 PM, all 7 days, in 22 local languages</div>
        <div style="font-weight:600; color:#0c4a75; margin-top:14px; margin-bottom:8px;">PM-KISAN Helpline</div>
        <div style="color:#333; font-size:0.95rem;">📱 Toll-free: <b>155261</b> / <b>1800-115-526</b> &nbsp;|&nbsp; ☎️ 011-24300606</div>
    </div>
""", unsafe_allow_html=True)
st.caption("Numbers verified from official Government of India sources. In the production version, this app will also link directly to the nearest Maharashtra Krishi Vibhag (Agriculture Department) extension officer based on the farmer's location.")

# ---------- Roadmap: expanding crop coverage ----------
st.markdown("### 🌱 Expanding crop coverage")
st.markdown("This prototype currently detects diseases in **tomato, potato, and bell pepper**. Given Maharashtra's major crops, we are actively expanding to:")

roadmap_col1, roadmap_col2 = st.columns(2)
with roadmap_col1:
    st.markdown("""
    - 🌾 **Jowar (sorghum)** — grain mold, downy mildew
    - 🌾 **Rice** — blast, bacterial leaf blight
    """)
with roadmap_col2:
    st.markdown("""
    - 🌿 **Cotton** — pink bollworm, leaf curl virus
    - 🎋 **Sugarcane** — red rot, smut
    """)
st.caption("These crops require additional field-image datasets for reliable detection — planned for the next development phase.")

st.markdown('<p class="footer-note">Prototype for SIH 2026 · Problem Statement SIH26131 · Government of Maharashtra</p>', unsafe_allow_html=True)    
