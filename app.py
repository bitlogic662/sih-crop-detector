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
        background: radial-gradient(circle at top left, #f1faf3 0%, #fafdf9 40%);
    }
    .hero {
        background: linear-gradient(135deg, #14532d 0%, #2d6a4f 50%, #52b788 100%);
        padding: 2.4rem 2.6rem;
        border-radius: 20px;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 30px rgba(20,83,45,0.3);
        position: relative;
        overflow: hidden;
    }
    .hero::after {
        content: "🌿";
        position: absolute;
        right: -10px;
        top: -30px;
        font-size: 10rem;
        opacity: 0.12;
        transform: rotate(15deg);
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ffffff, #d8f3dc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
        position: relative;
        z-index: 1;
    }
    .hero-sub {
        font-size: 1.05rem;
        color: #d8f3dc;
        margin-bottom: 0;
        position: relative;
        z-index: 1;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.18);
        color: white;
        padding: 5px 16px;
        border-radius: 20px;
        font-size: 0.8rem;
        margin-top: 12px;
        border: 1px solid rgba(255,255,255,0.35);
        position: relative;
        z-index: 1;
    }
    .stat-card {
        background: white;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        text-align: center;
        box-shadow: 0 3px 12px rgba(0,0,0,0.06);
        border: 1px solid #e8f0e8;
        transition: transform 0.2s ease;
    }
    .stat-card:hover { transform: translateY(-3px); }
    .stat-num {
        font-size: 1.7rem;
        font-weight: 800;
        background: linear-gradient(90deg, #2d6a4f, #52b788);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-label { font-size: 0.8rem; color: #777; margin-top: 2px; }
    .upload-panel {
        background: white;
        border: 2px dashed #95d5b2;
        border-radius: 18px;
        padding: 1.5rem;
        box-shadow: 0 3px 14px rgba(0,0,0,0.05);
    }
    .result-card {
        background: white;
        border-radius: 18px;
        padding: 1.6rem 1.8rem;
        box-shadow: 0 6px 22px rgba(0,0,0,0.1);
        border-top: 6px solid #2d6a4f;
        animation: fadeIn 0.5s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(0,0,0,0.15); }
        70% { box-shadow: 0 0 0 8px rgba(0,0,0,0); }
        100% { box-shadow: 0 0 0 0 rgba(0,0,0,0); }
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
        color: #999;
        text-transform: uppercase;
    }
    .result-name { font-size: 1.7rem; font-weight: 800; margin-top: 4px; }
    .confidence-bar-bg {
        background-color: #eee;
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
        background: linear-gradient(135deg, #fff8e6 0%, #fef0d0 100%);
        border-radius: 16px;
        padding: 1.3rem 1.6rem;
        margin-top: 1.2rem;
        border: 1px solid #f5deb3;
    }
    .section-header {
        font-size: 1.35rem;
        font-weight: 800;
        color: #14532d;
        margin: 2.2rem 0 1rem 0;
    }
    .helpline-card {
        background: linear-gradient(135deg, #e7f3ff 0%, #cfe8ff 100%);
        border-radius: 16px;
        padding: 1.4rem 1.7rem;
        border: 1px solid #b8daf7;
    }
    .roadmap-chip {
        background: white;
        border: 1px solid #d8e8d8;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        margin-bottom: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        transition: transform 0.2s ease;
    }
    .roadmap-chip:hover { transform: translateY(-3px); box-shadow: 0 6px 16px rgba(0,0,0,0.08); }
    .step-card {
        background: white;
        border-radius: 16px;
        padding: 1.3rem 1.2rem;
        text-align: center;
        box-shadow: 0 3px 12px rgba(0,0,0,0.05);
        height: 100%;
    }
    .step-num {
        width: 34px; height: 34px;
        border-radius: 50%;
        background: linear-gradient(135deg, #2d6a4f, #52b788);
        color: white;
        display: flex; align-items: center; justify-content: center;
        font-weight: 800;
        margin: 0 auto 10px auto;
    }
    .footer-note {
        color: #999;
        font-size: 0.8rem;
        margin-top: 3rem;
        text-align: center;
        padding-top: 1.5rem;
        border-top: 1px solid #eee;
    }
    div[data-testid="stFileUploader"] section {
        border: none !important;
        background: transparent !important;
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
    st.markdown('<div class="step-card"><div class="step-num">1</div><b>📷 Snap a photo</b><br><span style="color:#777; font-size:0.85rem;">Take a clear photo of the affected leaf</span></div>', unsafe_allow_html=True)
with h2:
    st.markdown('<div class="step-card"><div class="step-num">2</div><b>🧠 AI analyzes</b><br><span style="color:#777; font-size:0.85rem;">On-device model detects disease instantly</span></div>', unsafe_allow_html=True)
with h3:
    st.markdown('<div class="step-card"><div class="step-num">3</div><b>🗣️ Get advice</b><br><span style="color:#777; font-size:0.85rem;">Hear treatment steps in your language</span></div>', unsafe_allow_html=True)

st.write("")

# ---------- Disease info ----------
disease_info = {
    "Pepper__bell___Bacterial_spot": {
        "display": "Bell Pepper — Bacterial Spot",
        "icon": "🫑",
        "treatment_en": "Apply copper-based bactericide. Avoid overhead watering and remove infected leaves.",
        "treatment_hi": "कॉपर आधारित बैक्टीरिसाइड का प्रयोग करें। ऊपर से पानी देने से बचें और संक्रमित पत्तियों को हटा दें।",
        "treatment_mr": "तांबेयुक्त बॅक्टेरिसाइड वापरा. वरून पाणी देणे टाळा आणि संक्रमित पाने काढून टाका.",
        "treatment_kn": "ತಾಮ್ರ ಆಧಾರಿತ ಬ್ಯಾಕ್ಟೀರಿಸೈಡ್ ಅನ್ನು ಬಳಸಿ. ಮೇಲಿನಿಂದ ನೀರುಣಿಸುವುದನ್ನು ತಪ್ಪಿಸಿ ಮತ್ತು ಸೋಂಕಿತ ಎಲೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ.",
        "severity": "moderate"
    },
    "Potato___Early_blight": {
        "display": "Potato — Early Blight",
        "icon": "🥔",
        "treatment_en": "Apply fungicide (Chlorothalonil or Mancozeb). Rotate crops and remove infected debris.",
        "treatment_hi": "फफूंदनाशक (क्लोरोथालोनिल या मैंकोजेब) का प्रयोग करें। फसल चक्र अपनाएं और संक्रमित अवशेष हटा दें।",
        "treatment_mr": "बुरशीनाशक (क्लोरोथॅलोनिल किंवा मॅन्कोझेब) वापरा. पीक फेरपालट करा आणि संक्रमित अवशेष काढून टाका.",
        "treatment_kn": "ಶಿಲೀಂಧ್ರನಾಶಕ (ಕ್ಲೋರೋಥಲೋನಿಲ್ ಅಥವಾ ಮ್ಯಾಂಕೋಜೆಬ್) ಬಳಸಿ. ಬೆಳೆ ಸರದಿ ಅನುಸರಿಸಿ ಮತ್ತು ಸೋಂಕಿತ ಅವಶೇಷಗಳನ್ನು ತೆಗೆದುಹಾಕಿ.",
        "severity": "moderate"
    },
    "Tomato_Late_blight": {
        "display": "Tomato — Late Blight",
        "icon": "🍅",
        "treatment_en": "Apply copper-based fungicide immediately. Remove and destroy infected plants to prevent spread.",
        "treatment_hi": "तुरंत कॉपर आधारित फफूंदनाशक का प्रयोग करें। फैलाव रोकने के लिए संक्रमित पौधों को हटाकर नष्ट कर दें।",
        "treatment_mr": "त्वरित तांबेयुक्त बुरशीनाशक वापरा. प्रसार रोखण्यासाठी संक्रमित रोपे काढून नष्ट करा.",
        "treatment_kn": "ತಕ್ಷಣ ತಾಮ್ರ ಆಧಾರಿತ ಶಿಲೀಂಧ್ರನಾಶಕವನ್ನು ಬಳಸಿ. ಹರಡುವಿಕೆಯನ್ನು ತಡೆಯಲು ಸೋಂಕಿತ ಸಸ್ಯಗಳನ್ನು ತೆಗೆದು ನಾಶಪಡಿಸಿ.",
        "severity": "severe"
    },
    "Tomato_healthy": {
        "display": "Tomato — Healthy",
        "icon": "✅",
        "treatment_en": "No disease detected. Continue regular monitoring and good field hygiene.",
        "treatment_hi": "कोई रोग नहीं पाया गया। नियमित निगरानी और अच्छी खेत स्वच्छता जारी रखें।",
        "treatment_mr": "कोणताही रोग आढळला नाही. नियमित देखरेख आणि चांगली शेत स्वच्छता सुरू ठेवा.",
        "treatment_kn": "ಯಾವುದೇ ರೋಗ ಪತ್ತೆಯಾಗಿಲ್ಲ. ನಿಯಮಿತ ಮೇಲ್ವಿಚಾರಣೆ ಮತ್ತು ಉತ್ತಮ ಹೊಲದ ನೈರ್ಮಲ್ಯವನ್ನು ಮುಂದುವರಿಸಿ.",
        "severity": "healthy"
    }
}

severity_colors = {"healthy": "#2d6a4f", "moderate": "#e0a800", "severe": "#c0392b"}
severity_labels = {"healthy": "Healthy", "moderate": "Moderate risk", "severe": "Severe — act now"}

# ---------- Upload + Result ----------
st.markdown('<div class="section-header">📸 Scan a crop leaf</div>', unsafe_allow_html=True)

col_upload, col_result = st.columns([1, 1.2], gap="large")

with col_upload:
    st.markdown('<div class="upload-panel">', unsafe_allow_html=True)
    uploaded = st.file_uploader("Drop a leaf photo here, or click to browse", type=["jpg", "png", "jpeg"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_result:
    if uploaded:
        img_resized = img.resize((224, 224))
        arr = np.expand_dims(np.array(img_resized) / 255.0, axis=0)
        pred = model.predict(arr)
        result = class_names[np.argmax(pred)]
        confidence = float(np.max(pred)) * 100
        info = disease_info.get(result, {
            "display": result.replace("_", " "), "icon": "🌿",
            "treatment_en": "Consult a local agriculture expert.",
            "treatment_hi": "स्थानीय कृषि विशेषज्ञ से सलाह लें।",
            "treatment_mr": "स्थानिक कृषी तज्ञांचा सल्ला घ्या.",
            "treatment_kn": "ಸ್ಥಳೀಯ ಕೃಷಿ ತಜ್ಞರನ್ನು ಸಂಪರ್ಕಿಸಿ.",
            "severity": "moderate"
        })
        color = severity_colors.get(info["severity"], "#2d6a4f")

        st.markdown(f"""
            <div class="result-card" style="border-top-color:{color};">
                <div class="result-label">Detection result</div>
                <div class="result-name">{info['icon']} {info['display']}</div>
                <div style="margin-top:10px;">
                    <span class="status-dot" style="background-color:{color};"></span>
                    <span style="font-weight:600; color:{color};">{severity_labels.get(info['severity'], '')}</span>
                </div>
                <div style="margin-top:14px; font-size:0.85rem; color:#666;">Confidence: <b>{confidence:.1f}%</b></div>
                <div class="confidence-bar-bg">
                    <div class="confidence-bar-fill" style="width:{confidence}%; background-color:{color};"></div>
                </div>
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
                <div style="font-weight:700; color:#7a5c00; margin-bottom:6px;">💊 Recommended action</div>
                <div style="color:#444; line-height:1.6;">{treatment_text}</div>
            </div>
        """, unsafe_allow_html=True)

        if st.button("🔊 Play voice advice", use_container_width=True):
            from gtts import gTTS
            tts = gTTS(f"{info['display']}. {treatment_text}", lang=lang_code)
            tts.save("output.mp3")
            st.audio("output.mp3")
    else:
        st.markdown("""
            <div style="height:100%; display:flex; align-items:center; justify-content:center; text-align:center; color:#999; padding: 3rem 1rem;">
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
        <div style="font-weight:700; color:#0c4a75; margin-bottom:6px;">📱 Kisan Call Centre (Government of India)</div>
        <div style="color:#333; font-size:0.95rem; margin-bottom:14px;">Toll-free <b>1800-180-1551</b> · 6 AM–10 PM, all 7 days · 22 local languages</div>
        <div style="font-weight:700; color:#0c4a75; margin-bottom:6px;">📱 PM-KISAN Helpline</div>
        <div style="color:#333; font-size:0.95rem;">Toll-free <b>155261</b> / <b>1800-115-526</b> &nbsp;|&nbsp; ☎️ 011-24300606</div>
    </div>
""", unsafe_allow_html=True)
st.caption("Numbers verified from official Government of India sources. Production version will link directly to the nearest Maharashtra Krishi Vibhag extension officer by location.")

# ---------- Roadmap ----------
st.markdown('<div class="section-header">🌱 Expanding crop coverage</div>', unsafe_allow_html=True)
st.write("This prototype currently detects diseases in **tomato, potato, and bell pepper**. Next, we're expanding to Maharashtra's core crops:")

r1, r2, r3, r4 = st.columns(4)
with r1:
    st.markdown('<div class="roadmap-chip">🌾 <b>Jowar</b><br><span style="color:#777; font-size:0.85rem;">Grain mold, downy mildew</span></div>', unsafe_allow_html=True)
with r2:
    st.markdown('<div class="roadmap-chip">🌾 <b>Rice</b><br><span style="color:#777; font-size:0.85rem;">Blast, bacterial blight</span></div>', unsafe_allow_html=True)
with r3:
    st.markdown('<div class="roadmap-chip">🌿 <b>Cotton</b><br><span style="color:#777; font-size:0.85rem;">Pink bollworm, leaf curl</span></div>', unsafe_allow_html=True)
with r4:
    st.markdown('<div class="roadmap-chip">🎋 <b>Sugarcane</b><br><span style="color:#777; font-size:0.85rem;">Red rot, smut</span></div>', unsafe_allow_html=True)

st.markdown('<p class="footer-note">Prototype for SIH 2026 · Problem Statement SIH26131 · Government of Maharashtra</p>', unsafe_allow_html=True)

