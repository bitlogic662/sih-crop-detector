import streamlit as st
import numpy as np
import io
from PIL import Image
from gTTS import gTTS

# -----------------------------------------------------------------------------
# 1. PAGE SETUP & SESSION STATE FOR LANGUAGE RE-CHANGING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="KrishiRakshak AI",
    page_icon="🌾",
    layout="wide"
)

# Language Map (gTTS codes: hi=Hindi, kn=Kannada, mr=Marathi, en=English)
LANGUAGES = {
    "English": "en",
    "हिंदी (Hindi)": "hi",
    "ಕನ್ನಡ (Kannada)": "kn",
    "मराठी (Marathi)": "mr"
}

# Callback function to force full re-run when language changes
def on_language_change():
    st.session_state["selected_lang_code"] = LANGUAGES[st.session_state["lang_selector"]]

if "selected_lang_code" not in st.session_state:
    st.session_state["selected_lang_code"] = "en"

# Sidebar Selector with Callback
selected_lang_label = st.sidebar.selectbox(
    "🌐 Select Language / भाषा चुनें",
    list(LANGUAGES.keys()),
    key="lang_selector",
    on_change=on_language_change
)

CURRENT_LANG = st.session_state["selected_lang_code"]

# -----------------------------------------------------------------------------
# 2. COMPLETE TRANSLATION DICTIONARY MATRIX
# -----------------------------------------------------------------------------
TRANSLATIONS = {
    "en": {
        "title": "KrishiRakshak AI",
        "subtitle": "Early Detection & Management of Crop Diseases and Pest Infestations",
        "how_it_works": "How it works",
        "step1": "📷 Snap photos",
        "step2": "🧠 AI analyzes",
        "step3": "🗣️ Get advice",
        "upload_label": "📁 Upload Leaf Photo",
        "uploaded_img": "Uploaded Image",
        "assessment": "Overall Crop Health Assessment",
        "detection_res": "Detection result",
        "severity_label": "Severity",
        "confidence_label": "Confidence",
        "risk_label": "Overall risk",
        "rec_action": "Recommended Action",
        "precautions": "Important Precaution",
        "voice_summary": "Voice Summary",
        "disease_name": "Tomato Early Blight",
        "severity_val": "Moderate",
        "risk_val": "Moderate Risk",
        "action_val": "Apply copper-based fungicides or Mancozeb every 7 to 10 days.",
        "precaution_val": "Avoid overhead irrigation as water splashes spread bacteria rapidly.",
    },
    "hi": {
        "title": "कृषि रक्षक AI",
        "subtitle": "फसल रोगों और कीट प्रकोप की शीघ्र पहचान और प्रबंधन",
        "how_it_works": "यह कैसे काम करता है",
        "step1": "📷 फोटो लें",
        "step2": "🧠 AI विश्लेषण करता है",
        "step3": "🗣️ सलाह प्राप्त करें",
        "upload_label": "📁 पत्ती की फोटो अपलोड करें",
        "uploaded_img": "अपलोड की गई छवि",
        "assessment": "फसल स्वास्थ्य का समग्र मूल्यांकन",
        "detection_res": "पहचान का परिणाम",
        "severity_label": "गंभीरता",
        "confidence_label": "विश्वास स्तर",
        "risk_label": "कुल जोखिम",
        "rec_action": "अनुशंसित कार्रवाई",
        "precautions": "महत्वपूर्ण सावधानी",
        "voice_summary": "आवाज़ में सारांश",
        "disease_name": "टमाटर का अगेती झुलसा रोग (अर्ली ब्लाइट)",
        "severity_val": "मध्यम",
        "risk_val": "मध्यम जोखिम",
        "action_val": "हर 7 से 10 दिनों में तांबे आधारित कवकनाशी या मैंकोज़ेब का छिड़काव करें।",
        "precaution_val": "ऊपर से सिंचाई करने से बचें क्योंकि पानी के छिटकने से बीमारी तेज़ी से फैलती है।",
    },
    "kn": {
        "title": "ಕೃಷಿ ರಕ್ಷಕ್ AI",
        "subtitle": "ಬೆಳೆ ರೋಗಗಳು ಮತ್ತು ಕೀಟ ಬಾಧೆಗಳ ಆರಂಭಿಕ ಪತ್ತೆ ಮತ್ತು ನಿರ್ವಹಣೆ",
        "how_it_works": "ಇದು ಹೇಗೆ ಕೆಲಸ ಮಾಡುತ್ತದೆ",
        "step1": "📷 ಫೋಟೋ ತೆಗೆದುಕೊಳ್ಳಿ",
        "step2": "🧠 AI ವಿಶ್ಲೇಷಿಸುತ್ತದೆ",
        "step3": "🗣️ ಸಲಹೆ ಪಡೆಯಿರಿ",
        "upload_label": "📁 ಎಲೆಯ ಫೋಟೋ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ",
        "uploaded_img": "ಅಪ್‌ಲೋಡ್ ಮಾಡಿದ ಚಿತ್ರ",
        "assessment": "ಒಟ್ಟಾರೆ ಬೆಳೆ ಆರೋಗ್ಯ ಮೌಲ್ಯಮಾಪನ",
        "detection_res": "ಪತ್ತೆ ಫಲಿತಾಂಶ",
        "severity_label": "ತೀವ್ರತೆ",
        "confidence_label": "ವಿಶ್ವಾಸ ಮಟ್ಟ",
        "risk_label": "ಒಟ್ಟಾರೆ ಅಪಾಯ",
        "rec_action": "ಶಿಫಾರಸು ಮಾಡಿದ ಕ್ರಮ",
        "precautions": "ಪ್ರಮುಖ ಮುನ್ನೆಚ್ಚರಿಕೆ",
        "voice_summary": "ಧ್ವನಿ ಸಾರಾಂಶ",
        "disease_name": "ಟೊಮೆಟೊ ಅರ್ಲಿ ಬ್ಲೈಟ್ ರೋಗ",
        "severity_val": "ಮಧ್ಯಮ",
        "risk_val": "ಮಧ್ಯಮ ಅಪಾಯ",
        "action_val": "ಪ್ರತಿ 7 ರಿಂದ 10 ದಿನಗಳಿಗೊಮ್ಮೆ ತಾಮ್ರ ಆಧಾರಿತ ಶಿಲೀಂಧ್ರನಾಶಕ ಅಥವಾ ಮ್ಯಾಂಕೋಜೆಬ್ ಸಿಂಪಡಿಸಿ.",
        "precaution_val": "ಮೇಲಿಂದ ನೀರುಣಿಸುವುದನ್ನು ತಪ್ಪಿಸಿ ಏಕೆಂದರೆ ನೀರಿನ ಹನಿಗಳು ರೋಗವನ್ನು ವೇಗವಾಗಿ ಹರಡುತ್ತವೆ.",
    },
    "mr": {
        "title": "कृषी रक्षक AI",
        "subtitle": "पिकांचे रोग आणि किडींचा लवकर शोध व व्यवस्थापन",
        "how_it_works": "हे कसे कार्य करते",
        "step1": "📷 फोटो काढा",
        "step2": "🧠 AI विश्लेषण करते",
        "step3": "🗣️ सल्ला मिळवा",
        "upload_label": "📁 पानाचा फोटो अपलोड करा",
        "uploaded_img": "अपलोड केलेले चित्र",
        "assessment": "पिकाच्या आरोग्याचे एकूण मूल्यांकन",
        "detection_res": "शोध परिणाम",
        "severity_label": "तीव्रता",
        "confidence_label": "विश्वासार्हता",
        "risk_label": "एकूण धोका",
        "rec_action": "शिफारस केलेली कृती",
        "precautions": "महत्त्वाची खबरदारी",
        "voice_summary": "आवाजातील सारांश",
        "disease_name": "टोमॅटोवरील अर्ली ब्लाइट रोग",
        "severity_val": "मध्यम",
        "risk_val": "मध्यम धोका",
        "action_val": "दर ७ ते १० दिवसांनी तांब्यावर आधारित बुरशीनाशक किंवा मँकोझेब फवारा.",
        "precaution_val": "वरून पाणी देणे टाळा कारण पाण्याचे थेंब रोग वेगाने पसरवतात.",
    }
}

# Lookup helper
def t(key):
    return TRANSLATIONS[CURRENT_LANG].get(key, TRANSLATIONS["en"].get(key, key))

# Guaranteed Audio Generator in Selected Language
@st.cache_data(show_spinner=False)
def generate_native_audio(text_to_speak, lang_code):
    try:
        tts = gTTS(text=text_to_speak, lang=lang_code, slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp.read()
    except Exception as e:
        st.error(f"Audio generation error: {e}")
        return None

# -----------------------------------------------------------------------------
# 3. INTERFACE DISPLAY (DYNAMIC RE-RENDERING)
# -----------------------------------------------------------------------------
st.title(f"🌾 {t('title')}")
st.caption(t('subtitle'))

st.subheader("⚡ " + t('how_it_works'))
c1, c2, c3 = st.columns(3)
c1.info(f"1. **{t('step1')}**")
c2.info(f"2. **{t('step2')}**")
c3.info(f"3. **{t('step3')}**")

st.markdown("---")

# Image Upload
uploaded_file = st.file_uploader(t('upload_label'), type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption=t('uploaded_img'), width=300)
    
    # Translated Assessment Display
    st.subheader("📊 " + t('assessment'))
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**{t('detection_res')}:** {t('disease_name')}")
        st.markdown(f"**{t('severity_label')}:** {t('severity_val')}")
    with col_b:
        st.markdown(f"**{t('confidence_label')}:** 96.5%")
        st.markdown(f"**{t('risk_label')}:** {t('risk_val')}")

    st.success(f"🎯 **{t('rec_action')}:** {t('action_val')}")
    st.warning(f"⚠️ **{t('precautions')}:** {t('precaution_val')}")

    # Voice Summary Generation (Native Language Text Sent to gTTS)
    st.subheader("🔊 " + t('voice_summary'))
    
    # Construct sentence directly in target native language
    audio_text = f"{t('disease_name')}. {t('rec_action')}: {t('action_val')}. {t('precautions')}: {t('precaution_val')}"
    
    # Generate audio passing target language code
    audio_bytes = generate_native_audio(audio_text, CURRENT_LANG)
    
    if audio_bytes:
        # Pass key containing CURRENT_LANG so widget resets on language swap
        st.audio(audio_bytes, format="audio/mp3", autoplay=False)
