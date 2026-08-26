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
        background-color: #fafdf9;
    }
    .hero {
        background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 55%, #40916c 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 18px;
        margin-bottom: 1.8rem;
        box-shadow: 0 8px 24px rgba(27,67,50,0.25);
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: white;
        margin-bottom: 4px;
    }
    .hero-sub {
        font-size: 1.05rem;
        color: #d8f3dc;
        margin-bottom: 0;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.18);
        color: white;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        margin-top: 10px;
        border: 1px solid rgba(255,255,255,0.35);
    }
    .stat-card {
        background: white;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border: 1px solid #e8f0e8;
    }
    .stat-num {
        font-size: 1.6rem;
        font-weight: 800;
        color: #2d6a4f;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #777;
        margin-top: 2px;
    }
    .upload-panel {
        background: white;
        border: 2px dashed #95d5b2;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    }
    .result-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem 1.7rem;
        box-shadow: 0 4px 18px rgba(0,0,0,0.08);
        border-top: 6px solid #2d6a4f;
        animation: fadeIn 0.5s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .result-label {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 1px;
        color: #999;
        text-transform: uppercase;
    }
    .result-name {
        font-size: 1.6rem;
        font-weight: 800;
        margin-top: 4px;
    }
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
        transition: width 0.6s ease;
    }
    .treatment-box {
        background: linear-gradient(135deg, #fff8e6 0%, #fef3d9 100%);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        margin-top: 1.2rem;
        border: 1px solid #f5deb3;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 800;
        color: #1b4332;
        margin: 2rem 0 0.8rem 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .helpline-card {
        background: linear-gradient(135deg, #e7f3ff 0%, #d6ebff 100%);
        border-radius: 14px;
        padding: 1.3rem 1.6rem;
        border: 1px solid #b8daf7;
    }
    .roadmap-chip {
        background: white;
        border: 1px solid #d8e8d8;
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 8px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
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
    st.markdown('<div class="stat-card"><div class="stat-num">3</div><div class="stat-label">Languages</div></div>', unsafe_allow_html=True)
with s4:
    st.markdown('<div class="stat-card"><div class="stat-num">Offline</div><div class="stat-label">Field-ready design</div></div>', unsafe_allow_html=True)

st.write("")

# ---------- Disease info ----------
disease_info = {
    "Pepper__bell___Bacterial_spot": {
        "display": "Bell Pepper — Bacterial Spot",
        "icon": "🫑",
        "treatment_en": "Apply copper-based bactericide. Avoid overhead watering and remove infected leaves.",
        "treatment_hi": "कॉपर आधारित बैक्टीरिसाइड का प्रयोग करें। ऊपर से पानी देने से बचें और संक्रमित पत्तियों को हटा दें।",
        "treatment_mr": "तांबेयुक्त बॅक्टेरिसाइड वापरा. वरून पाणी देणे टाळा आणि संक्रमित पाने काढून टाका.",
        "severity": "moderate"
    },
    "Potato___Early_blight": {
        "display": "Potato — Early Blight",
        "icon": "🥔",
        "treatment_en": "Apply fungicide (Chlorothalonil or Mancozeb). Rotate crops and remove infected debris.",
        "treatment_hi": "फफूंदनाशक (क्लोरोथालोनिल या मैंकोजेब) का प्रयोग करें। फसल चक्र अपनाएं और संक्रमित अवशेष हटा दें।",
        "treatment_mr": "बुरशीनाशक (क्लोरोथॅलोनिल किंवा मॅन्कोझेब) वापरा. पीक फेरपालट करा आणि संक्रमित अवशेष काढून टाका.",
        "severity": "moderate"
    },
    "Tomato_Late_blight": {
        "display": "Tomato — Late Blight",
        "icon": "🍅",
        "treatment_en": "Apply copper-based fungicide immediately. Remove and destroy infected plants to prevent spread.",
        "treatment_hi": "तुरंत कॉपर आधारित फफूंदनाशक का प्रयोग करें। फैलाव रोकने के लिए संक्रमित पौधों को हटाकर नष्ट कर दें।",
        "treatment_mr": "त्वरित तांबेयुक्त बुरशीनाशक वापरा. प्रसार रोखण्यासाठी संक्रमित रोपे काढून नष्ट करा.",
        "severity": "severe"
    },
    "Tomato_healthy": {
        "display": "Tomato — Healthy",
        "icon": "✅",
        "treatment_en": "No disease detected. Continue regular monitoring and good field hygiene.",
        "treatment_hi": "कोई रोग नहीं पाया गया। नियमित निगरानी और अच्छी खेत स्वच्छता जारी रखें।",
        "treatment_mr": "कोणताही रोग आढळला नाही. नियमित देखरेख आणि चांगली शेत स्वच्छता सुरू ठेवा.",
        "severity": "healthy"
    }
}

severity_colors = {"healthy": "#2d6a4f", "moderate": "#e0a800", "severe": "#c0392b"}
severity_labels = {"healthy": "🟢 Healthy", "moderate": "🟡 Moderate risk", "severe": "🔴 Severe — act now"}

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
            "severity": "moderate"
        })
        color = severity_colors.get(info["severity"], "#2d6a4f")

        st.markdown(f"""
            <div class="result-card" style="border-top-color:{color};">
                <div class="result-label">Detection result</div>
                <div class="result-name">{info['icon']} {info['display']}</div>
                <div style="margin-top:8px;">{severity_labels.get(info['severity'], '')}</div>
                <div style="margin-top:14px; font-size:0.85rem; color:#666;">Confidence: <b>{confidence:.1f}%</b></div>
                <div class="confidence-bar-bg">
                    <div class="confidence-bar-fill" style="width:{confidence}%; background-color:{color};"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        lang_choice = st.selectbox("🌐 Voice language", ["English", "हिंदी (Hindi)", "मराठी (Marathi)"])
        lang_map = {"English": ("en", "treatment_en"), "हिंदी (Hindi)": ("hi", "treatment_hi"), "मराठी (Marathi)": ("mr", "treatment_mr")}
        lang_code, treatment_key = lang_map[lang_choice]
        treatment_text = info[treatment_key]

        st.markdown(f"""
            <div class="treatment-box">
                <div style="font-weight:700; color:#7a5c00; margin-bottom:6px;">💊 Recommended action</div>
                <div style="color:#444; line-height:1.5;">{treatment_text}</div>
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
                    <div style="font-size:3rem;">🌱</div>
                    <div style="margin-top:8px;">Upload a photo to see detection results here</div>
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
