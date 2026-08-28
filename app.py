import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageStat
import numpy as np
import json
import io
from collections import Counter
from gtts import gTTS

st.set_page_config(
    page_title="KrishiRakshak AI - Multi-Language Crop Advisory",
    page_icon="🌾",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 1. LANGUAGE & DICTIONARY CONFIGURATION
# -----------------------------------------------------------------------------
LANGUAGES = {
    "English": "en",
    "हिंदी (Hindi)": "hi",
    "मराठी (Marathi)": "mr",
    "ಕನ್ನಡ (Kannada)": "kn",
}

# Sidebar Language Selector
st.sidebar.markdown("### 🌐 Select Language / भाषा चुनें")
selected_language = st.sidebar.selectbox(
    "Select language",
    list(LANGUAGES.keys()),
    key="global_language_selector",
    label_visibility="collapsed"
)
CURRENT_LANG = LANGUAGES[selected_language]

# Translation Dictionary Matrix
UI_TRANSLATIONS = {
    "en": {},
    "hi": {
        # App UI Header & Sections
        "KrishiRakshak AI": "कृषि रक्षक AI",
        "Early Detection & Management of Crop Diseases and Pest Infestations": "फसल रोगों और कीट प्रकोप की शीघ्र पहचान और प्रबंधन",
        "How it works": "यह कैसे काम करता है",
        "Snap photos": "📷 फोटो लें",
        "AI analyzes": "🧠 AI विश्लेषण करता है",
        "Get advice": "🗣️ सलाह प्राप्त करें",
        "Input Method": "इनपुट का तरीका",
        "📷 Click Leaf Photo": "📷 पत्ती की फोटो लें",
        "📁 Upload Leaf Photo": "📁 पत्ती की फोटो अपलोड करें",
        "Scan crop leaves": "फसल की पत्तियों को स्कैन करें",
        "Crop Information": "फसल की जानकारी",
        "Crop name": "फसल का नाम",
        "Crop growth stage": "फसल की वृद्धि अवस्था",
        "Recent weather / field condition": "हाल की मौसम स्थिति",
        "Analyze All Photos": "सभी तस्वीरों का विश्लेषण करें",
        
        # Results & Metrics
        "Individual Photo Results": "व्यक्तिगत फोटो परिणाम",
        "Overall Crop Health Assessment": "फसल स्वास्थ्य का समग्र मूल्यांकन",
        "Detection result": "पहचान का परिणाम",
        "Recommended Action": "अनुशंसित कार्रवाई",
        "Important Precaution": "महत्वपूर्ण सावधानी",
        "Confidence": "विश्वास स्तर",
        "Severity": "गंभीरता",
        "Voice Summary": "आवाज़ में सारांश",
        "Overall risk": "कुल जोखिम",
        "Low Risk": "कम जोखिम",
        "Moderate Risk": "मध्यम जोखिम",
        "High Risk": "उच्च जोखिम",
        "Healthy": "स्वस्थ",
        "Moderate": "मध्यम",
        "Severe": "गंभीर",
        
        # Disease Names & Treatments
        "Tomato Early Blight": "टमाटर का अगेती झुलसा (अर्ली ब्लाइट)",
        "Potato Late Blight": "आलू का पछेती झुलसा (लेट ब्लाइट)",
        "Tomato Bacterial Spot": "टमाटर का बैक्टीरियल धब्बा रोग",
        "Healthy Leaf": "स्वस्थ पत्ती",
        "Apply copper-based fungicides or Mancozeb every 7-10 days.": "हर 7-10 दिनों में तांबे आधारित कवकनाशी या मैंकोज़ेब का छिड़काव करें।",
        "Apply systemic fungicides like Ridomil Gold immediately.": "तुरंत रिडोमिल गोल्ड जैसे प्रणालीगत कवकनाशी का प्रयोग करें।",
        "Avoid overhead irrigation as water splashes spread bacteria rapidly.": "ऊपर से सिंचाई करने से बचें क्योंकि पानी के छिटकने से जीवाणु तेज़ी से फैलते हैं।"
    },
    "mr": {
        "KrishiRakshak AI": "कृषी रक्षक AI",
        "Early Detection & Management of Crop Diseases and Pest Infestations": "पिकांचे रोग आणि किडींचा लवकर शोध व व्यवस्थापन",
        "How it works": "हे कसे कार्य करते",
        "Snap photos": "📷 फोटो काढा",
        "AI analyzes": "🧠 AI विश्लेषण करते",
        "Get advice": "🗣️ सल्ला मिळवा",
        "Input Method": "इनपुट पद्धत",
        "📷 Click Leaf Photo": "📷 पानाचा फोटो काढा",
        "📁 Upload Leaf Photo": "📁 पानाचा फोटो अपलोड करा",
        "Scan crop leaves": "पिकाची पाने स्कॅन करा",
        "Crop Information": "पिकाची माहिती",
        "Crop name": "पिकाचे नाव",
        "Crop growth stage": "पिकाच्या वाढीची अवस्था",
        "Recent weather / field condition": "हवामान / शेताची स्थिती",
        "Analyze All Photos": "सर्व फोटोंचे विश्लेषण करा",
        "Individual Photo Results": "वैयक्तिक फोटो परिणाम",
        "Overall Crop Health Assessment": "पिकाच्या आरोग्याचे एकूण मूल्यांकन",
        "Detection result": "शोध परिणाम",
        "Recommended Action": "शिफारस केलेली कृती",
        "Important Precaution": "महत्त्वाची खबरदारी",
        "Confidence": "विश्वास",
        "Severity": "तीव्रता",
        "Voice Summary": "आवाजातील सारांश",
        "Overall risk": "एकूण धोका",
        "Low Risk": "कमी धोका",
        "Moderate Risk": "मध्यम धोका",
        "High Risk": "जास्त धोका",
        "Healthy": "निरोगी",
        "Moderate": "मध्यम",
        "Severe": "गंभीर",
        "Tomato Early Blight": "टोमॅटोवरील अर्ली ब्लाइट रोग",
        "Potato Late Blight": "बटाट्यावरील लेट ब्लाइट रोग",
        "Tomato Bacterial Spot": "टोमॅटोवरील बॅक्टेरियल स्पॉट",
        "Healthy Leaf": "निरोगी पान",
        "Apply copper-based fungicides or Mancozeb every 7-10 days.": "दर ७-१० दिवसांनी तांब्यावर आधारित बुरशीनाशक किंवा मँकोझेब फवारा.",
        "Apply systemic fungicides like Ridomil Gold immediately.": "रिडोमिल गोल्डसारख्या बुरशीनाशकाचा त्वरित वापर करा.",
        "Avoid overhead irrigation as water splashes spread bacteria rapidly.": "वरून पाणी देणे टाळा कारण पाण्याचे थेंब बॅक्टेरिया वेगाने पसरवतात."
    },
    "kn": {
        "KrishiRakshak AI": "ಕೃಷಿ ರಕ್ಷಕ್ AI",
        "Early Detection & Management of Crop Diseases and Pest Infestations": "ಬೆಳೆ ರೋಗಗಳು ಮತ್ತು ಕೀಟ ಬಾಧೆಗಳ ಆರಂಭಿಕ ಪತ್ತೆ ಮತ್ತು ನಿರ್ವಹಣೆ",
        "How it works": "ಇದು ಹೇಗೆ ಕೆಲಸ ಮಾಡುತ್ತದೆ",
        "Snap photos": "📷 ಫೋಟೋ ತೆಗೆದುಕೊಳ್ಳಿ",
        "AI analyzes": "🧠 AI ವಿಶ್ಲೇಷಿಸುತ್ತದೆ",
        "Get advice": "🗣️ ಸಲಹೆ ಪಡೆಯಿರಿ",
        "Input Method": "ಇನ್‌ಪುಟ್ ವಿಧಾನ",
        "📷 Click Leaf Photo": "📷 ಎಲೆಯ ಫೋಟೋ ತೆಗೆದುಕೊಳ್ಳಿ",
        "📁 Upload Leaf Photo": "📁 ಎಲೆಯ ಫೋಟೋ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ",
        "Scan crop leaves": "ಬೆಳೆ ಎಲೆಗಳನ್ನು ಸ್ಕ್ಯಾನ್ ಮಾಡಿ",
        "Crop Information": "ಬೆಳೆ ಮಾಹಿತಿ",
        "Crop name": "ಬೆಳೆಯ ಹೆಸರು",
        "Crop growth stage": "ಬೆಳೆಯ ಬೆಳವಣಿಗೆಯ ಹಂತ",
        "Recent weather / field condition": "ಇತ್ತೀಚಿನ ಹವಾಮಾನ ಪರಿಸ್ಥಿತಿ",
        "Analyze All Photos": "ಎಲ್ಲಾ ಫೋಟೋಗಳನ್ನು ವಿಶ್ಲೇಷಿಸಿ",
        "Individual Photo Results": "ವೈಯಕ್ತಿಕ ಫೋಟೋ ಫಲಿತಾಂಶಗಳು",
        "Overall Crop Health Assessment": "ಒಟ್ಟಾರೆ ಬೆಳೆ ಆರೋಗ್ಯ ಮೌಲ್ಯಮಾಪನ",
        "Detection result": "ಪತ್ತೆ ಫಲಿತಾಂಶ",
        "Recommended Action": "ಶಿಫಾರಸು ಮಾಡಿದ ಕ್ರಮ",
        "Important Precaution": "ಪ್ರಮುಖ ಮುನ್ನೆಚ್ಚರಿಕೆ",
        "Confidence": "ವಿಶ್ವಾಸ",
        "Severity": "ತೀವ್ರತೆ",
        "Voice Summary": "ಧ್ವನಿ ಸಾರಾಂಶ",
        "Overall risk": "ಒಟ್ಟಾರೆ ಅಪಾಯ",
        "Low Risk": "ಕಡಿಮೆ ಅಪಾಯ",
        "Moderate Risk": "ಮಧ್ಯಮ ಅಪಾಯ",
        "High Risk": "ಹೆಚ್ಚಿನ ಅಪಾಯ",
        "Healthy": "ಆರೋಗ್ಯಕರ",
        "Moderate": "ಮಧ್ಯಮ",
        "Severe": "ತೀವ್ರ",
        "Tomato Early Blight": "ಟೊಮೆಟೊ ಅರ್ಲಿ ಬ್ಲೈಟ್ ರೋಗ",
        "Potato Late Blight": "ಆಲೂಗಡ್ಡೆ ಲೇಟ್ ಬ್ಲೈಟ್ ರೋಗ",
        "Tomato Bacterial Spot": "ಟೊಮೆಟೊ ಬ್ಯಾಕ್ಟೀರಿಯಲ್ ಸ್ಪಾಟ್",
        "Healthy Leaf": "ಆರೋಗ್ಯಕರ ಎಲೆ",
        "Apply copper-based fungicides or Mancozeb every 7-10 days.": "ಪ್ರತಿ 7-10 ದಿನಗಳಿಗೊಮ್ಮೆ ತಾಮ್ರ ಆಧಾರಿತ ಶಿಲೀಂಧ್ರನಾಶಕ ಅಥವಾ ಮ್ಯಾಂಕೋಜೆಬ್ ಸಿಂಪಡಿಸಿ.",
        "Apply systemic fungicides like Ridomil Gold immediately.": "ತಕ್ಷಣವೇ ರಿಡೋಮಿಲ್ ಗೋಲ್ಡ್ ನಂತಹ ಸಿಸ್ಟಮಿಕ್ ಫಂಗಿಸೈಡ್ ಬಳಸಿ.",
        "Avoid overhead irrigation as water splashes spread bacteria rapidly.": "ಮೇಲಿಂದ ನೀರುಣಿಸುವುದನ್ನು ತಪ್ಪಿಸಿ ಏಕೆಂದರೆ ನೀರಿನ ಹನಿಗಳು ಬ್ಯಾಕ್ಟೀರಿಯಾವನ್ನು ವೇಗವಾಗಿ ಹರಡುತ್ತವೆ."
    }
}

# Centralized Translator Helper
def translate(text):
    """Looks up and translates UI strings into the currently selected language."""
    if not isinstance(text, str) or CURRENT_LANG == "en" or not text.strip():
        return text
    table = UI_TRANSLATIONS.get(CURRENT_LANG, {})
    return table.get(text, text)

# Text-To-Speech Converter
def generate_audio(text_content):
    """Generates an MP3 audio file in the selected language."""
    try:
        tts = gTTS(text=text_content, lang=CURRENT_LANG, slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp.read()
    except Exception:
        return None

# -----------------------------------------------------------------------------
# 2. SAMPLE DISEASE DATABASE (IN ENGLISH BASE KEYS)
# -----------------------------------------------------------------------------
disease_db = {
    "Tomato_Early_blight": {
        "name": "Tomato Early Blight",
        "severity": "Moderate",
        "action": "Apply copper-based fungicides or Mancozeb every 7-10 days.",
        "precaution": "Avoid overhead irrigation as water splashes spread bacteria rapidly."
    },
    "Potato_Late_blight": {
        "name": "Potato Late Blight",
        "severity": "Severe",
        "action": "Apply systemic fungicides like Ridomil Gold immediately.",
        "precaution": "Destroy severely infected plants and maintain field sanitation."
    },
    "Healthy": {
        "name": "Healthy Leaf",
        "severity": "Healthy",
        "action": "No treatment needed. Continue standard agricultural care.",
        "precaution": "Inspect regularly for early signs of pests."
    }
}

# -----------------------------------------------------------------------------
# 3. INTERFACE RENDER
# -----------------------------------------------------------------------------
st.title(f"🌾 {translate('KrishiRakshak AI')}")
st.caption(translate("Early Detection & Management of Crop Diseases and Pest Infestations"))

st.subheader("⚡ " + translate("How it works"))
col1, col2, col3 = st.columns(3)
col1.info(f"1. **{translate('Snap photos')}**")
col2.info(f"2. **{translate('AI analyzes')}**")
col3.info(f"3. **{translate('Get advice')}**")

# Input Section
st.subheader("📸 " + translate("Scan crop leaves"))
upload_file = st.file_uploader(translate("📁 Upload Leaf Photo"), type=["jpg", "jpeg", "png"])

if upload_file is not None:
    img = Image.open(upload_file)
    st.image(img, caption=translate("Uploaded Image"), width=300)
    
    # Mock Detection (Replace with your Tensorflow model inference)
    detected_key = "Tomato_Early_blight" 
    disease = disease_db[detected_key]

    # Translate Outputs Dynamically
    translated_name = translate(disease["name"])
    translated_severity = translate(disease["severity"])
    translated_action = translate(disease["action"])
    translated_precaution = translate(disease["precaution"])

    # Render Translated Diagnosis
    st.markdown("---")
    st.subheader("📊 " + translate("Overall Crop Health Assessment"))
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**{translate('Detection result')}:** {translated_name}")
        st.markdown(f"**{translate('Severity')}:** {translated_severity}")
    with c2:
        st.markdown(f"**{translate('Confidence')}:** 96.5%")
        st.markdown(f"**{translate('Overall risk')}:** {translate('Moderate Risk')}")

    st.success(f"🎯 **{translate('Recommended Action')}:** {translated_action}")
    st.warning(f"⚠️ **{translate('Important Precaution')}:** {translated_precaution}")

    # Generate Dynamic Voice Advice
    st.subheader("🔊 " + translate("Voice Summary"))
    full_audio_text = f"{translated_name}. {translate('Severity')}: {translated_severity}. {translated_action}"
    audio_data = generate_audio(full_audio_text)
    
    if audio_data:
        st.audio(audio_data, format="audio/mp3")
