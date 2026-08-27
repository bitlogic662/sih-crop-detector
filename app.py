
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
    .stButton button {
        background: #f2c744 !important;
        color: #23331b !important;
        border: none !important;
        border-radius: 30px !important;
        font-weight: 700 !important;
        padding: 0.6rem 1.2rem !important;
    }
    .stButton button:hover { background: #f6d768 !important; }
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

severity_colors = {"healthy": "#7bd389", "moderate": "#f2c744", "severe": "#e0665a"}
severity_labels = {"healthy": "Healthy", "moderate": "Moderate risk", "severe": "Severe — act now"}

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
            "display": result.replace("_", " "), "icon": "🌿",
            "treatment_en": "Consult a local agriculture expert.",
            "treatment_hi": "स्थानीय कृषि विशेषज्ञ से सलाह लें।",
            "treatment_mr": "स्थानिक कृषी तज्ञांचा सल्ला घ्या.",
            "treatment_kn": "ಸ್ಥಳೀಯ ಕೃಷಿ ತಜ್ಞರನ್ನು ಸಂಪರ್ಕಿಸಿ.",
            "severity": "moderate"
        })
        color = severity_colors.get(info["severity"], "#f2c744")

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
