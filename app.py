import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, UnidentifiedImageError, ImageStat
import numpy as np
import json
import io
from collections import Counter
from gtts import gTTS

st.set_page_config(
    page_title="KrishiRakshak AI - Crop Disease Detection",
    page_icon="🌾",
    layout="wide"
)

# Configuration settings
CONFIDENCE_THRESHOLD = 60.0  # Percentage minimum required for a valid diagnosis

# ---------- MULTI-LANGUAGE UI SUPPORT ----------
LANGUAGES = {
    "English": "en",
    "हिंदी (Hindi)": "hi",
    "मराठी (Marathi)": "mr",
    "ಕನ್ನಡ (Kannada)": "kn",
}

UI_TRANSLATIONS = {
    "en": {},
    "hi": {
        "Early Detection & Management of Crop Diseases and Pest Infestations": "फसल रोगों और कीट प्रकोप की शीघ्र पहचान और प्रबंधन",
        "Crops covered": "कवर की गई फसलें",
        "Model accuracy": "मॉडल की सटीकता",
        "Languages": "भाषाएँ",
        "AI Advisory": "एआई सलाह",
        "Field-ready design": "खेत के लिए उपयुक्त डिज़ाइन",
        "How it works": "यह कैसे काम करता है",
        "Snap photos": "📷 फोटो लें",
        "Take clear photos of the affected leaves": "प्रभावित पत्तियों की साफ तस्वीरें लें",
        "AI analyzes": "🧠 AI विश्लेषण करता है",
        "Model detects disease instantly": "मॉडल तुरंत रोग का पता लगाता है",
        "Get advice": "🗣️ सलाह प्राप्त करें",
        "Hear treatment steps in your language": "अपनी भाषा में उपचार के चरण सुनें",
        "Input Method": "इनपुट का तरीका",
        "📷 Click Leaf Photo": "📷 पत्ती की फोटो लें",
        "📁 Upload Leaf Photo": "📁 पत्ती की फोटो अपलोड करें",
        "Capture Leaf Photo": "पत्ती की फोटो लें",
        "Take a clear, well-lit photo of the affected crop leaf": "प्रभावित फसल की पत्ती की एक स्पष्ट और अच्छी रोशनी वाली तस्वीर लें",
        "Scan crop leaves": "फसल की पत्तियाँ स्कैन करें",
        "Drag one or more leaf photos below, or click to browse": "नीचे एक या अधिक पत्तियों की तस्वीरें डालें या ब्राउज़ करने के लिए क्लिक करें",
        "Crop Information": "फसल की जानकारी",
        "Crop name": "फसल का नाम",
        "Crop growth stage": "फसल की वृद्धि अवस्था",
        "Not sure about crop age": "फसल की आयु के बारे में निश्चित नहीं",
        "Approximate crop age (days)": "फसल की अनुमानित आयु (दिन)",
        "How long have you noticed the symptoms?": "आपने लक्षण कितने समय से देखे हैं?",
        "How much of the crop appears affected?": "फसल का कितना हिस्सा प्रभावित दिखाई देता है?",
        "Recent weather / field condition": "हाल की मौसम / खेत की स्थिति",
        "Have you already applied any treatment?": "क्या आपने पहले से कोई उपचार किया है?",
        "Please specify the treatment used": "कृपया इस्तेमाल किए गए उपचार का विवरण दें",
        "Has this crop shown this disease before? (optional)": "क्या इस फसल में पहले भी यह रोग हुआ है? (वैकल्पिक)",
        "How many times has this crop shown this disease before?": "इस फसल में यह रोग पहले कितनी बार हुआ है?",
        "Type of soil used for growing": "उगाने के लिए इस्तेमाल की गई मिट्टी का प्रकार",
        "Village / City (optional)": "गाँव / शहर (वैकल्पिक)",
        "District (optional)": "जिला (वैकल्पिक)",
        "Analyze All Photos": "सभी तस्वीरों का विश्लेषण करें",
        "Individual Photo Results": "व्यक्तिगत फोटो परिणाम",
        "Overall Crop Health Assessment": "फसल स्वास्थ्य का समग्र मूल्यांकन",
        "Farmer Information": "किसान की जानकारी",
        "Detailed Recommendation": "विस्तृत सुझाव",
        "Recommended Action": "अनुशंसित कार्रवाई",
        "Treatment": "उपचार",
        "Weather Risk": "मौसम का जोखिम",
        "How Quickly Should You Act?": "आपको कितनी जल्दी कार्रवाई करनी चाहिए?",
        "Important Precaution": "महत्वपूर्ण सावधानी",
        "Voice Summary": "आवाज़ में सारांश",
        "Farmer helpline & support": "किसान हेल्पलाइन और सहायता",
        "Expanding crop coverage": "फसल कवरेज का विस्तार",
        "Listen to this result": "इस परिणाम को सुनें",
        "Listen to full recommendation summary": "पूरी अनुशंसा सुनें",
        "Photos analyzed": "विश्लेषित तस्वीरें",
        "Affected photos": "प्रभावित तस्वीरें",
        "Healthy photos": "स्वस्थ तस्वीरें",
        "Avg. confidence": "औसत विश्वास स्तर",
        "Overall risk": "कुल जोखिम",
        "Confidence": "विश्वास स्तर",
        "Severity": "गंभीरता",
        "Progression": "प्रगति",
        "Crop health": "फसल स्वास्थ्य",
        "Healthy": "स्वस्थ",
        "Moderate risk": "मध्यम जोखिम",
        "Severe — act now": "गंभीर — अभी कार्रवाई करें",
        "No disease detected": "कोई रोग नहीं पाया गया",
        "Low Risk": "कम जोखिम",
        "Moderate Risk": "मध्यम जोखिम",
        "High Risk": "उच्च जोखिम",
        "Preventive care": "रोकथाम संबंधी देखभाल",
        "Act immediately": "तुरंत कार्रवाई करें",
        "Act within 1–2 days": "1–2 दिनों के भीतर कार्रवाई करें",
        "Monitor closely": "ध्यान से निगरानी करें",
        "No": "नहीं",
        "Yes": "हाँ",
        "Not sure": "निश्चित नहीं",
        "Normal": "सामान्य",
        "High rainfall": "अधिक वर्षा",
        "High humidity": "अधिक आर्द्रता",
        "Very hot": "बहुत गर्म",
        "Very dry": "बहुत शुष्क",
        "Less than 1 day": "1 दिन से कम",
        "1–3 days": "1–3 दिन",
        "4–7 days": "4–7 दिन",
        "1–2 weeks": "1–2 सप्ताह",
        "More than 2 weeks": "2 सप्ताह से अधिक",
        "Only one/few leaves": "केवल एक/कुछ पत्तियाँ",
        "Less than 25%": "25% से कम",
        "25–50%": "25–50%",
        "50–75%": "50–75%",
        "More than 75%": "75% से अधिक",
        "Tomato": "टमाटर",
        "Potato": "आलू",
        "Bell Pepper": "शिमला मिर्च",
        "Other / Not sure": "अन्य / निश्चित नहीं",
        "Seedling": "अंकुर अवस्था",
        "Vegetative": "वानस्पतिक अवस्था",
        "Flowering": "फूल अवस्था",
        "Fruiting": "फल अवस्था",
        "Mature": "परिपक्व अवस्था",
        "Loamy soil": "दोमट मिट्टी",
        "Clayey soil": "चिकनी मिट्टी",
        "Sandy soil": "बलुई मिट्टी",
        "Black soil (Regur)": "काली मिट्टी (रेगुर)",
        "Red soil": "लाल मिट्टी",
        "Alluvial soil": "जलोढ़ मिट्टी",
        "Please capture a photo first.": "कृपया पहले एक तस्वीर लें।",
        "Please upload at least one photo first.": "कृपया पहले कम से कम एक फोटो अपलोड करें।",
        "Analyzing photo(s)...": "तस्वीरों का विश्लेषण किया जा रहा है...",
        "Photo": "फोटो",
        "Detection result": "पहचान का परिणाम",
        "Unsupported image": "असमर्थित छवि",
        "⚠️ Invalid Image: This image does not appear to be a supported crop leaf. Please upload or capture a clear photo of a supported crop leaf.": "❌ यह तस्वीर समर्थित फसल की पत्ती की तस्वीर नहीं लगती। कृपया समर्थित फसल की पत्ती की स्पष्ट तस्वीर अपलोड करें या लें।",
        "Image resolution is too low. Please upload a clearer photo.": "तस्वीर का रिज़ॉल्यूशन बहुत कम है। कृपया एक स्पष्ट फोटो अपलोड करें।",
        "Image appears too blurry or lacks visible detail. Please provide a sharp photo.": "तस्वीर बहुत धुंधली लगती है या विवरण की कमी है। कृपया एक स्पष्ट फोटो प्रदान करें।",
        "The model is not confident enough in this image prediction. Please ensure it is a clear leaf photo.": "मॉडल इस तस्वीर के पूर्वानुमान के प्रति आश्वस्त नहीं है। कृपया सुनिश्चित करें कि यह पत्ती की स्पष्ट फोटो है।",
        "No valid crop leaf images were found among the inputs provided.": "प्रदान किए गए इनपुट में कोई वैध फसल पत्ती की छवियां नहीं मिलीं।",
        "None of the photos provided were valid crop leaf images suitable for disease analysis.": "प्रदान की गई कोई भी फोटो बीमारी के विश्लेषण के लिए उपयुक्त वैध फसल की पत्ती नहीं थी।",
        "Invalid or Corrupted Image": "अवैध या दूषित छवि",
        "Failed to load image file.": "इमेज फ़ाइल लोड करने में विफल।",
        "AI predictions may vary based on photo quality. Always consult a local agricultural officer or expert for major crop decisions.": "तस्वीर की गुणवत्ता के आधार पर AI पूर्वानुमान भिन्न हो सकते हैं। मुख्य फसल निर्णयों के लिए हमेशा स्थानीय कृषि अधिकारी या विशेषज्ञ से परामर्श लें।",
        "Kisan Call Centre (Toll-Free): 1800-180-1551": "किसान कॉल सेंटर (टोल-फ्री): 1800-180-1551",
        "Krishi Vigyan Kendra (KVK) Network": "कृषि विज्ञान केंद्र (केवीके) नेटवर्क",
        "Department of Agriculture, Maharashtra": "कृषि विभाग, महाराष्ट्र",
        "Current model covers Tomato, Potato, and Bell Pepper leaf diseases.": "वर्तमान मॉडल टमाटर, आलू और शिमला मिर्च के पत्तों के रोगों को कवर करता है।",
        "Expanding to Sugarcane, Cotton, Soybean, and Rice in upcoming versions.": "आगामी संस्करणों में गन्ना, कपास, सोयाबीन और चावल तक विस्तार किया जा रहा है।",
        "Location": "स्थान",
        "Growth Stage": "वृद्धि की अवस्था",
        "Soil": "मिट्टी",
        "Symptoms Duration": "लक्षणों की अवधि",
        "Crop Affected": "प्रभावित फसल",
        "Prior History": "पूर्व इतिहास",
        "Previous Treatment": "पिछला उपचार",
        "Yes, applied:": "हाँ, लागू किया गया:",
        "Times seen before:": "पहले देखे जाने की संख्या:",
        "Select language": "भाषा चुनें",
        "Working...": "काम जारी है...",
        "None": "कोई नहीं",
        "Unknown": "अज्ञात",
        "Low": "कम",
        "Moderate": "मध्यम",
        "High": "उच्च",
        "Severe": "गंभीर",
        "Early": "शुरुआती", "Advanced": "उन्नत", "Critical": "गंभीर",
        "Early stage": "शुरुआती अवस्था", "Progressed": "प्रगति पर", "Advanced stage": "उन्नत अवस्था"
    },
    "mr": {
        "Early Detection & Management of Crop Diseases and Pest Infestations": "पिकांचे रोग आणि किडींचा लवकर शोध व व्यवस्थापन",
        "Crops covered": "समाविष्ट पिके",
        "Model accuracy": "मॉडेल अचूकता",
        "Languages": "भाषा",
        "AI Advisory": "एआय सल्ला",
        "Field-ready design": "शेतासाठी तयार डिझाइन",
        "How it works": "हे कसे कार्य करते",
        "Snap photos": "📷 फोटो काढा",
        "Take clear photos of the affected leaves": "प्रभावित पानांचे स्पष्ट फोटो काढा",
        "AI analyzes": "🧠 AI विश्लेषण करते",
        "Model detects disease instantly": "मॉडेल रोगाचा त्वरित शोध घेते",
        "Get advice": "🗣️ सल्ला मिळवा",
        "Hear treatment steps in your language": "आपल्या भाषेत उपचाराच्या सूचना ऐका",
        "Input Method": "इनपुट पद्धत",
        "📷 Click Leaf Photo": "📷 पानाचा फोटो काढा",
        "📁 Upload Leaf Photo": "📁 पानाचा फोटो अपलोड करा",
        "Capture Leaf Photo": "पानाचा फोटो काढा",
        "Take a clear, well-lit photo of the affected crop leaf": "प्रभावित पिकाच्या पानाचा स्पष्ट आणि चांगल्या प्रकाशातील फोटो काढा",
        "Scan crop leaves": "पिकांची पाने स्कॅन करा",
        "Drag one or more leaf photos below, or click to browse": "खाली एक किंवा अधिक पानांचे फोटो टाका किंवा ब्राउझ करण्यासाठी क्लिक करा",
        "Crop Information": "पिकाची माहिती",
        "Crop name": "पिकाचे नाव",
        "Crop growth stage": "पिकाची वाढीची अवस्था",
        "Not sure about crop age": "पिकाच्या वयाबद्दल खात्री नाही",
        "Approximate crop age (days)": "पिकाचे अंदाजे वय (दिवस)",
        "How long have you noticed the symptoms?": "लक्षणे किती दिवसांपासून दिसत आहेत?",
        "How much of the crop appears affected?": "पिकाचा किती भाग प्रभावित दिसतो?",
        "Recent weather / field condition": "अलीकडील हवामान / शेताची स्थिती",
        "Have you already applied any treatment?": "तुम्ही आधीच काही उपचार केले आहेत का?",
        "Please specify the treatment used": "वापरलेल्या उपचाराचे नाव द्या",
        "Has this crop shown this disease before? (optional)": "या पिकाला यापूर्वी हा रोग झाला आहे का? (पर्यायी)",
        "How many times has this crop shown this disease before?": "या पिकाला हा रोग यापूर्वी किती वेळा झाला आहे?",
        "Type of soil used for growing": "पिकासाठी वापरलेल्या मातीचा प्रकार",
        "Village / City (optional)": "गाव / शहर (पर्यायी)",
        "District (optional)": "जिल्हा (पर्यायी)",
        "Analyze All Photos": "सर्व फोटोंचे विश्लेषण करा",
        "Individual Photo Results": "वैयक्तिक फोटो परिणाम",
        "Overall Crop Health Assessment": "पिकाच्या आरोग्याचे एकूण मूल्यांकन",
        "Farmer Information": "शेतकऱ्याची माहिती",
        "Detailed Recommendation": "सविस्तर शिफारस",
        "Recommended Action": "शिफारस केलेली कृती",
        "Treatment": "उपचार",
        "Weather Risk": "हवामानाचा धोका",
        "How Quickly Should You Act?": "किती लवकर कृती करावी?",
        "Important Precaution": "महत्त्वाची खबरदारी",
        "Voice Summary": "आवाजातील सारांश",
        "Farmer helpline & support": "शेतकरी हेल्पलाइन आणि मदत",
        "Expanding crop coverage": "पिकांचा विस्तार",
        "Listen to this result": "हा परिणाम ऐका",
        "Listen to full recommendation summary": "संपूर्ण शिफारस ऐका",
        "Photos analyzed": "विश्लेषित फोटो",
        "Affected photos": "प्रभावित फोटो",
        "Healthy photos": "निरोगी फोटो",
        "Avg. confidence": "सरासरी विश्वास",
        "Overall risk": "एकूण धोका",
        "Confidence": "विश्वास",
        "Severity": "तीव्रता",
        "Progression": "प्रगती",
        "Crop health": "पिकाचे आरोग्य",
        "Healthy": "निरोगी",
        "Moderate risk": "मध्यम धोका",
        "Severe — act now": "गंभीर — त्वरित कृती करा",
        "No disease detected": "कोणताही रोग आढळला नाही",
        "Low Risk": "कमी धोका",
        "Moderate Risk": "मध्यम धोका",
        "High Risk": "जास्त धोका",
        "Preventive care": "प्रतिबंधात्मक काळजी",
        "Act immediately": "त्वरित कृती करा",
        "Act within 1–2 days": "1–2 दिवसांत कृती करा",
        "Monitor closely": "लक्षपूर्वक निरीक्षण करा",
        "No": "नाही",
        "Yes": "होय",
        "Not sure": "खात्री नाही",
        "Normal": "सामान्य",
        "High rainfall": "जास्त पाऊस",
        "High humidity": "जास्त आर्द्रता",
        "Very hot": "खूप उष्ण",
        "Very dry": "खूप कोरडे",
        "Less than 1 day": "1 दिवसापेक्षा कमी",
        "1–3 days": "1–3 दिवस",
        "4–7 days": "4–7 दिवस",
        "1–2 weeks": "1–2 आठवडे",
        "More than 2 weeks": "2 आठवड्यांपेक्षा जास्त",
        "Only one/few leaves": "फक्त एक/काही पाने",
        "Less than 25%": "25% पेक्षा कमी",
        "25–50%": "25–50%",
        "50–75%": "50–75%",
        "More than 75%": "75% पेक्षा जास्त",
        "Tomato": "टोमॅटो",
        "Potato": "बटाटा",
        "Bell Pepper": "ढोबळी मिरची",
        "Other / Not sure": "इतर / खात्री नाही",
        "Seedling": "रोप अवस्था",
        "Vegetative": "शाकीय अवस्था",
        "Flowering": "फुलोरा अवस्था",
        "Fruiting": "फळधारणा अवस्था",
        "Mature": "परिपक्व अवस्था",
        "Loamy soil": "गाळाची माती",
        "Clayey soil": "चिकणमाती",
        "Sandy soil": "वालुकामय माती",
        "Black soil (Regur)": "काळी माती (रेगूर)",
        "Red soil": "लाल माती",
        "Alluvial soil": "गाळाची माती",
        "Please capture a photo first.": "कृपया प्रथम फोटो काढा.",
        "Please upload at least one photo first.": "कृपया आधी किमान एक फोटो अपलोड करा.",
        "Analyzing photo(s)...": "फोटोंचे विश्लेषण करत आहे...",
        "Photo": "फोटो",
        "Detection result": "शोध परिणाम",
        "Unsupported image": "असमर्थित फोटो",
        "⚠️ Invalid Image: This image does not appear to be a supported crop leaf. Please upload or capture a clear photo of a supported crop leaf.": "❌ हा फोटो समर्थित पिकाच्या पानाचा फोटो दिसत नाही. कृपया समर्थित पिकाच्या पानाचा स्पष्ट फोटो अपलोड करा किंवा काढा.",
        "Image resolution is too low. Please upload a clearer photo.": "फोटोचे रिझोल्यूशन खूप कमी आहे. कृपया अधिक स्पष्ट फोटो अपलोड करा.",
        "Image appears too blurry or lacks visible detail. Please provide a sharp photo.": "फोटो खूप अस्पष्ट दिसतो किंवा त्यात तपशील कमी आहेत. कृपया स्पष्ट फोटो द्या.",
        "The model is not confident enough in this image prediction. Please ensure it is a clear leaf photo.": "मॉडेलला या फोटोच्या अंदाजाबद्दल पुरेशी खात्री नाही. कृपया तो पानाचा स्पष्ट फोटो असल्याची खात्री करा.",
        "No valid crop leaf images were found among the inputs provided.": "दिलेल्या इनपुटमध्ये कोणतेही वैध पिकाचे पान आढळले नाही.",
        "None of the photos provided were valid crop leaf images suitable for disease analysis.": "दिलेल्या फोटोंपैकी एकही फोटो रोग विश्लेषणासाठी योग्य पिकाच्या पानाचा नव्हता.",
        "Invalid or Corrupted Image": "अवैध किंवा खराब झालेला फोटो",
        "Failed to load image file.": "इमेज फाइल लोड करण्यात अपयशी.",
        "AI predictions may vary based on photo quality. Always consult a local agricultural officer or expert for major crop decisions.": "फोटोच्या गुणवत्तेनुसार AI अंदाज बदलू शकतात. पिकाच्या महत्त्वाच्या निर्णयांसाठी नेहमी स्थानिक कृषी अधिकारी किंवा तज्ञांचा सल्ला घ्या.",
        "Kisan Call Centre (Toll-Free): 1800-180-1551": "किसान कॉल सेंटर (टोल-फ्री): 1800-180-1551",
        "Krishi Vigyan Kendra (KVK) Network": "कृषी विज्ञान केंद्र (KVK) नेटवर्क",
        "Department of Agriculture, Maharashtra": "कृषी विभाग, महाराष्ट्र",
        "Current model covers Tomato, Potato, and Bell Pepper leaf diseases.": "सध्याचे मॉडेल टोमॅटो, बटाटा आणि ढोबळी मिरचीच्या पानांच्या रोगांचा समावेश करते.",
        "Expanding to Sugarcane, Cotton, Soybean, and Rice in upcoming versions.": "येत्या आवृत्त्यांमध्ये ऊस, कापूस, सोयाबीन आणि भात पिकांपर्यंत विस्तार करत आहोत.",
        "Location": "स्थान",
        "Growth Stage": "वाढीची अवस्था",
        "Soil": "माती",
        "Symptoms Duration": "लक्षणांचा कालावधी",
        "Crop Affected": "प्रभावित पीक",
        "Prior History": "मागील इतिहास",
        "Previous Treatment": "मागील उपचार",
        "Yes, applied:": "होय, वापरले:",
        "Times seen before:": "पूर्वी पाहिल्याची संख्या:",
        "Select language": "भाषा निवडा",
        "Working...": "प्रक्रिया सुरू आहे...",
        "None": "काहीही नाही",
        "Unknown": "अज्ञात",
        "Low": "कमी",
        "Moderate": "मध्यम",
        "High": "उच्च",
        "Severe": "गंभीर",
        "Early": "शुरुवातीचे", "Advanced": "प्रगत", "Critical": "गंभीर",
        "Early stage": "शुरुवातीची अवस्था", "Progressed": "प्रगतीपथावर", "Advanced stage": "प्रगत अवस्था"
    },
    "kn": {
        "Early Detection & Management of Crop Diseases and Pest Infestations": "ಬೆಳೆ ರೋಗಗಳು ಮತ್ತು ಕೀಟ ಬಾಧೆಗಳ ಆರಂಭಿಕ ಪತ್ತೆ ಮತ್ತು ನಿರ್ವಹಣೆ",
        "Crops covered": "ಒಳಗೊಂಡ ಬೆಳೆಗಳು",
        "Model accuracy": "ಮಾದರಿ ನಿಖರತೆ",
        "Languages": "ಭಾಷೆಗಳು",
        "AI Advisory": "ಎಐ ಸಲಹೆ",
        "Field-ready design": "ಹೊಲಕ್ಕೆ ಸಿದ್ಧ ವಿನ್ಯಾಸ",
        "How it works": "ಇದು ಹೇಗೆ ಕೆಲಸ ಮಾಡುತ್ತದೆ",
        "Snap photos": "📷 ಫೋಟೋ ತೆಗೆದುಕೊಳ್ಳಿ",
        "Take clear photos of the affected leaves": "ಬಾಧಿತ ಎಲೆಗಳ ಸ್ಪಷ್ಟ ಫೋಟೋಗಳನ್ನು ತೆಗೆದುಕೊಳ್ಳಿ",
        "AI analyzes": "🧠 AI ವಿಶ್ಲೇಷಿಸುತ್ತದೆ",
        "Model detects disease instantly": "ಮಾದರಿಯು ರೋಗವನ್ನು ತಕ್ಷಣ ಪತ್ತೆಹಚ್ಚುತ್ತದೆ",
        "Get advice": "🗣️ ಸಲಹೆ ಪಡೆಯಿರಿ",
        "Hear treatment steps in your language": "ನಿಮ್ಮ ಭಾಷೆಯಲ್ಲಿ ಚಿಕಿತ್ಸಾ ಕ್ರಮಗಳನ್ನು ಕೇಳಿ",
        "Input Method": "ಇನ್‌ಪುಟ್ ವಿಧಾನ",
        "📷 Click Leaf Photo": "📷 ಎಲೆಯ ಫೋಟೋ ತೆಗೆದುಕೊಳ್ಳಿ",
        "📁 Upload Leaf Photo": "📁 ಎಲೆಯ ಫೋಟೋ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ",
        "Capture Leaf Photo": "ಎಲೆಯ ಫೋಟೋ ತೆಗೆದುಕೊಳ್ಳಿ",
        "Take a clear, well-lit photo of the affected crop leaf": "ಬಾಧಿತ ಬೆಳೆ ಎಲೆಯ ಸ್ಪಷ್ಟ ಮತ್ತು ಬೆಳಕಿರುವ ಫೋಟೋ ತೆಗೆದುಕೊಳ್ಳಿ",
        "Scan crop leaves": "ಬೆಳೆ ಎಲೆಗಳನ್ನು ಸ್ಕ್ಯಾನ್ ಮಾಡಿ",
        "Drag one or more leaf photos below, or click to browse": "ಕೆಳಗೆ ಒಂದು ಅಥವಾ ಹೆಚ್ಚಿನ ಎಲೆಗಳ ಫೋಟೋಗಳನ್ನು ಹಾಕಿ ಅಥವಾ ಬ್ರೌಸ್ ಮಾಡಲು ಕ್ಲಿಕ್ ಮಾಡಿ",
        "Crop Information": "ಬೆಳೆ ಮಾಹಿತಿ",
        "Crop name": "ಬೆಳೆಯ ಹೆಸರು",
        "Crop growth stage": "ಬೆಳೆಯ ಬೆಳವಣಿಗೆಯ ಹಂತ",
        "Not sure about crop age": "ಬೆಳೆಯ ವಯಸ್ಸಿನ ಬಗ್ಗೆ ಖಚಿತವಿಲ್ಲ",
        "Approximate crop age (days)": "ಬೆಳೆಯ ಅಂದಾಜು ವಯಸ್ಸು (ದಿನಗಳು)",
        "How long have you noticed the symptoms?": "ರೋಗಲಕ್ಷಣಗಳು ಎಷ್ಟು ಸಮಯದಿಂದ ಕಾಣಿಸುತ್ತಿವೆ?",
        "How much of the crop appears affected?": "ಬೆಳೆಯ ಎಷ್ಟು ಭಾಗ ಬಾಧಿತವಾಗಿದೆ?",
        "Recent weather / field condition": "ಇತ್ತೀಚಿನ ಹವಾಮಾನ / ಹೊಲದ ಪರಿಸ್ಥಿತಿ",
        "Have you already applied any treatment?": "ನೀವು ಈಗಾಗಲೇ ಯಾವುದೇ ಚಿಕಿತ್ಸೆ ನೀಡಿದ್ದೀರಾ?",
        "Please specify the treatment used": "ಬಳಸಿದ ಚಿಕಿತ್ಸೆಯನ್ನು ನಮೂದಿಸಿ",
        "Has this crop shown this disease before? (optional)": "ಈ ಬೆಳೆಗೆ ಈ ರೋಗವು ಹಿಂದೆ ಕಾಣಿಸಿಕೊಂಡಿದೆಯೇ? (ಐಚ್ಛಿಕ)",
        "How many times has this crop shown this disease before?": "ಈ ಬೆಳೆಗೆ ಈ ರೋಗವು ಹಿಂದೆ ಎಷ್ಟು ಬಾರಿ ಕಾಣಿಸಿಕೊಂಡಿದೆ?",
        "Type of soil used for growing": "ಬೆಳೆಯಲು ಬಳಸಿದ ಮಣ್ಣಿನ ವಿಧ",
        "Village / City (optional)": "ಗ್ರಾಮ / ನಗರ (ಐಚ್ಛಿಕ)",
        "District (optional)": "ಜಿಲ್ಲೆ (ಐಚ್ಛಿಕ)",
        "Analyze All Photos": "ಎಲ್ಲಾ ಫೋಟೋಗಳನ್ನು ವಿಶ್ಲೇಷಿಸಿ",
        "Individual Photo Results": "ವೈಯಕ್ತಿಕ ಫೋಟೋ ಫಲಿತಾಂಶಗಳು",
        "Overall Crop Health Assessment": "ಒಟ್ಟಾರೆ ಬೆಳೆ ಆರೋಗ್ಯ ಮೌಲ್ಯಮಾಪನ",
        "Farmer Information": "ರೈತರ ಮಾಹಿತಿ",
        "Detailed Recommendation": "ವಿವರವಾದ ಶಿಫಾರಸು",
        "Recommended Action": "ಶಿಫಾರಸು ಮಾಡಿದ ಕ್ರಮ",
        "Treatment": "ಚಿಕಿತ್ಸೆ",
        "Weather Risk": "ಹವಾಮಾನ ಅಪಾಯ",
        "How Quickly Should You Act?": "ಎಷ್ಟು ಬೇಗ ಕ್ರಮ ಕೈಗೊಳ್ಳಬೇಕು?",
        "Important Precaution": "ಪ್ರಮುಖ ಮುನ್ನೆಚ್ಚರಿಕೆ",
        "Voice Summary": "ಧ್ವನಿ ಸಾರಾಂಶ",
        "Farmer helpline & support": "ರೈತರ ಸಹಾಯವಾಣಿ ಮತ್ತು ಬೆಂಬಲ",
        "Expanding crop coverage": "ಬೆಳೆ ವ್ಯಾಪ್ತಿಯ ವಿಸ್ತರಣೆ",
        "Listen to this result": "ಈ ಫಲಿತಾಂಶವನ್ನು ಆಲಿಸಿ",
        "Listen to full recommendation summary": "ಸಂಪೂರ್ಣ ಶಿಫಾರಸನ್ನು ಆಲಿಸಿ",
        "Photos analyzed": "ವಿಶ್ಲೇಷಿಸಿದ ಫೋಟೋಗಳು",
        "Affected photos": "ಬಾಧಿತ ಫೋಟೋಗಳು",
        "Healthy photos": "ಆರೋಗ್ಯಕರ ಫೋಟೋಗಳು",
        "Avg. confidence": "ಸರಾಸರಿ ವಿಶ್ವಾಸ",
        "Overall risk": "ಒಟ್ಟಾರೆ ಅಪಾಯ",
        "Confidence": "ವಿಶ್ವಾಸ",
        "Severity": "ತೀವ್ರತೆ",
        "Progression": "ಪ್ರಗತಿ",
        "Crop health": "ಬೆಳೆ ಆರೋಗ್ಯ",
        "Healthy": "ಆರೋಗ್ಯಕರ",
        "Moderate risk": "ಮಧ್ಯಮ ಅಪಾಯ",
        "Severe — act now": "ತೀವ್ರ — ಈಗಲೇ ಕ್ರಮ ಕೈಗೊಳ್ಳಿ",
        "No disease detected": "ಯಾವುದೇ ರೋಗ ಪತ್ತೆಯಾಗಿಲ್ಲ",
        "Low Risk": "ಕಡಿಮೆ ಅಪಾಯ",
        "Moderate Risk": "ಮಧ್ಯಮ ಅಪಾಯ",
        "High Risk": "ಹೆಚ್ಚಿನ ಅಪಾಯ",
        "Preventive care": "ತಡೆಗಟ್ಟುವ ಆರೈಕೆ",
        "Act immediately": "ತಕ್ಷಣ ಕ್ರಮ ಕೈಗೊಳ್ಳಿ",
        "Act within 1–2 days": "1–2 ದಿನಗಳಲ್ಲಿ ಕ್ರಮ ಕೈಗೊಳ್ಳಿ",
        "Monitor closely": "ನಿಕಟವಾಗಿ ಮೇಲ್ವಿಚಾರಣೆ ಮಾಡಿ",
        "No": "ಇಲ್ಲ",
        "Yes": "ಹೌದು",
        "Not sure": "ಖಚಿತವಿಲ್ಲ",
        "Normal": "ಸಾಮಾನ್ಯ",
        "High rainfall": "ಹೆಚ್ಚಿನ ಮಳೆ",
        "High humidity": "ಹೆಚ್ಚಿನ ತೇವಾಂಶ",
        "Very hot": "ತುಂಬಾ ಬಿಸಿ",
        "Very dry": "ತುಂಬಾ ಒಣ",
        "Less than 1 day": "1 ದಿನಕ್ಕಿಂತ ಕಡಿಮೆ",
        "1–3 days": "1–3 ದಿನಗಳು",
        "4–7 days": "4–7 ದಿನಗಳು",
        "1–2 weeks": "1–2 ವಾರಗಳು",
        "More than 2 weeks": "2 ವಾರಗಳಿಗಿಂತ ಹೆಚ್ಚು",
        "Only one/few leaves": "ಒಂದು/ಕೆಲವು ಎಲೆಗಳು ಮಾತ್ರ",
        "Less than 25%": "25% ಕ್ಕಿಂತ ಕಡಿಮೆ",
        "25–50%": "25–50%",
        "50–75%": "50–75%",
        "More than 75%": "75% ಕ್ಕಿಂತ ಹೆಚ್ಚು",
        "Tomato": "ಟೊಮೇಟೊ",
        "Potato": "ಆಲೂಗಡ್ಡೆ",
        "Bell Pepper": "ದೊಡ್ಡ ಮೆಣಸಿನಕಾಯಿ",
        "Other / Not sure": "ಇತರೆ / ಖಚಿತವಿಲ್ಲ",
        "Seedling": "ಸಸಿ ಹಂತ",
        "Vegetative": "ಸಸ್ಯೀಯ ಹಂತ",
        "Flowering": "ಹೂ ಬಿಡುವ ಹಂತ",
        "Fruiting": "ಹಣ್ಣು ಬಿಡುವ ಹಂತ",
        "Mature": "ಪಕ್ವ ಹಂತ",
        "Loamy soil": "ಲೋಮಿ ಮಣ್ಣು",
        "Clayey soil": "ಜೇಡಿ ಮಣ್ಣು",
        "Sandy soil": "ಮರಳು ಮಣ್ಣು",
        "Black soil (Regur)": "ಕಪ್ಪು ಮಣ್ಣು (ರೆಗರ್)",
        "Red soil": "ಕೆಂಪು ಮಣ್ಣು",
        "Alluvial soil": "ಜಲೋಢ ಮಣ್ಣು",
        "Please capture a photo first.": "ದಯವಿಟ್ಟು ಮೊದಲು ಫೋಟೋ ತೆಗೆಯಿರಿ.",
        "Please upload at least one photo first.": "ದಯವಿಟ್ಟು ಮೊದಲು ಕನಿಷ್ಠ ಒಂದು ಫೋಟೋವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ.",
        "Analyzing photo(s)...": "ಫೋಟೋಗಳನ್ನು ವಿಶ್ಲೇಷಿಸಲಾಗುತ್ತಿದೆ...",
        "Photo": "ಫೋಟೋ",
        "Detection result": "ಪತ್ತೆ ಫಲಿತಾಂಶ",
        "Unsupported image": "ಬೆಂಬಲವಿಲ್ಲದ ಚಿತ್ರ",
        "⚠️ Invalid Image: This image does not appear to be a supported crop leaf. Please upload or capture a clear photo of a supported crop leaf.": "❌ ಈ ಚಿತ್ರವು ಬೆಂಬಲಿತ ಬೆಳೆ ಎಲೆಯ ಚಿತ್ರವಾಗಿರುವಂತೆ ಕಾಣುತ್ತಿಲ್ಲ. ದಯವಿಟ್ಟು ಬೆಂಬಲಿತ ಬೆಳೆ ಎಲೆಯ ಸ್ಪಷ್ಟ ಫೋಟೋವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ ಅಥವಾ ತೆಗೆದುಕೊಳ್ಳಿ.",
        "Image resolution is too low. Please upload a clearer photo.": "ಚಿತ್ರದ রেজಲ್ಯೂಶನ್ ತುಂಬಾ ಕಡಿಮೆಯಾಗಿದೆ. ದಯವಿಟ್ಟು ಸ್ಪಷ್ಟವಾದ ಫೋಟೋವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ.",
        "Image appears too blurry or lacks visible detail. Please provide a sharp photo.": "ಚಿತ್ರವು ತುಂಬಾ ಮಸುಕಾಗಿ ಕಾಣುತ್ತದೆ ಅಥವಾ ವಿವರಗಳ ಕೊರತೆಯಿದೆ. ದಯವಿಟ್ಟು ಸ್ಪಷ್ಟವಾದ ಫೋಟೋವನ್ನು ನೀಡಿ.",
        "The model is not confident enough in this image prediction. Please ensure it is a clear leaf photo.": "ಈ ಚಿತ್ರದ ಮುನ್ನೋಟದ ಬಗ್ಗೆ ಮಾದರಿಗೆ ಸಾಕಷ್ಟು ವಿಶ್ವಾಸವಿಲ್ಲ. ದಯವಿಟ್ಟು ಇದು ಸ್ಪಷ್ಟ ಎಲೆಯ ಫೋಟೋ ಎಂದು ಖಚಿತಪಡಿಸಿಕೊಳ್ಳಿ.",
        "No valid crop leaf images were found among the inputs provided.": "ನೀಡಿದ ಇನ್‌ಪುಟ್‌ಗಳಲ್ಲಿ ಯಾವುದೇ ಸಿಂಧು ಬೆಳೆ ಎಲೆಯ ಚಿತ್ರಗಳು ಕಂಡುಬಂದಿಲ್ಲ.",
        "None of the photos provided were valid crop leaf images suitable for disease analysis.": "ನೀಡಿದ ಯಾವುದೇ ಫೋಟೋಗಳು ರೋಗ ವಿಶ್ಲೇಷಣೆಗೆ ಸೂಕ್ತವಾದ ಬೆಳೆ ಎಲೆಯ ಫೋಟೋಗಳಾಗಿರಲಿಲ್ಲ.",
        "Invalid or Corrupted Image": "ಅಮಾನ್ಯ ಅಥವಾ ಹಾಳಾದ ಚಿತ್ರ",
        "Failed to load image file.": "ಚಿತ್ರ ಫೈಲ್ ಲೋಡ್ ಮಾಡಲು ವಿಫಲವಾಗಿದೆ.",
        "AI predictions may vary based on photo quality. Always consult a local agricultural officer or expert for major crop decisions.": "ಫೋಟೋ ಗುಣಮಟ್ಟವನ್ನು ಆಧರಿಸಿ AI ಮುನ್ನೋಟಗಳು ಬದಲಾಗಬಹುದು. ಪ್ರಮುಖ ಬೆಳೆ ನಿರ್ಧಾರಗಳಿಗಾಗಿ ಯಾವಾಗಲೂ ಸ್ಥಳೀಯ ಕೃಷಿ ಅಧಿಕಾರಿ ಅಥವಾ ತಜ್ಞರನ್ನು ಸಂಪರ್ಕಿಸಿ.",
        "Kisan Call Centre (Toll-Free): 1800-180-1551": "ರೈತ ಕರೆ ಕೇಂದ್ರ (ಉಚಿತ ಕರೆ): 1800-180-1551",
        "Krishi Vigyan Kendra (KVK) Network": "ಕೃಷಿ ವಿಜ್ಞಾನ ಕೇಂದ್ರ (KVK) ಜಾಲ",
        "Department of Agriculture, Maharashtra": "ಕೃಷಿ ಇಲಾಖೆ, ಮಹಾರಾಷ್ಟ್ರ",
        "Current model covers Tomato, Potato, and Bell Pepper leaf diseases.": "ಪ್ರಸ್ತುತ ಮಾದರಿಯು ಟೊಮೆಟೊ, ಆಲೂಗಡ್ಡೆ ಮತ್ತು ದೊಡ್ಡ ಮೆಣಸಿನಕಾಯಿ ಎಲೆ ರೋಗಗಳನ್ನು ಒಳಗೊಂಡಿದೆ.",
        "Expanding to Sugarcane, Cotton, Soybean, and Rice in upcoming versions.": "ರಾಬರುವ ಆವೃತ್ತಿಗಳಲ್ಲಿ ಕಬ್ಬು, ಹತ್ತಿ, ಸೋಯಾಬೀನ್ ಮತ್ತು ಭತ್ತಕ್ಕೆ ವಿಸ್ತರಿಸಲಾಗುತ್ತಿದೆ.",
        "Location": "ಸ್ಥಳ",
        "Growth Stage": "ಬೆಳವಣಿಗೆಯ ಹಂತ",
        "Soil": "ಮಣ್ಣು",
        "Symptoms Duration": "ರೋಗಲಕ್ಷಣಗಳ ಅವಧಿ",
        "Crop Affected": "ಬಾಧಿತ ಬೆಳೆ",
        "Prior History": "ಹಿಂದಿನ ಇತಿಹಾಸ",
        "Previous Treatment": "ಹಿಂದಿನ ಚಿಕಿತ್ಸೆ",
        "Yes, applied:": "ಹೌದು, ನೀಡಲಾಗಿದೆ:",
        "Times seen before:": "ಹಿಂದೆ ಕಂಡ ಸಮಯಗಳು:",
        "Select language": "ಭಾಷೆಯನ್ನು ಆಯ್ಕೆಮಾಡಿ",
        "Working...": "ಕೆಲಸ ನಡೆಯುತ್ತಿದೆ...",
        "None": "ಯಾವುದೂ ಇಲ್ಲ",
        "Unknown": "ಅಜ್ಞಾತ",
        "Low": "ಕಡಿಮೆ",
        "Moderate": "ಮಧ್ಯಮ",
        "High": "ಹೆಚ್ಚು",
        "Severe": "ತೀವ್ರ",
        "Early": "ಆರಂಭಿಕ", "Advanced": "ಸುಧಾರಿತ", "Critical": "ತೀವ್ರ",
        "Early stage": "ಆರಂಭಿಕ ಹಂತ", "Progressed": "ಪ್ರಗತಿಯಲ್ಲಿದೆ", "Advanced stage": "ಸುಧಾರಿತ ಹಂತ"
    }
}

# Sidebar Language Selector
st.sidebar.markdown("### 🌐 Language / भाषा / भाषा / ಭಾಷೆ")

# Always the full, fixed set of supported languages (English, Hindi, Marathi,
# Kannada). Rebuilt fresh every run from LANGUAGES so no language — English
# included — can ever be dropped from the list across repeated switches.
_language_options = list(LANGUAGES.keys())

# Guard against a stale/invalid value ever being left in session_state (e.g.
# from a previous version of this app) so the widget always has a valid,
# explicit index to select — this is what keeps switching back and forth
# (English -> Kannada -> English -> Marathi -> English -> ...) reliable.
if st.session_state.get("global_language_selector") not in _language_options:
    st.session_state["global_language_selector"] = _language_options[0]

selected_language = st.sidebar.selectbox(
    "Select language",
    _language_options,
    index=_language_options.index(st.session_state["global_language_selector"]),
    key="global_language_selector",
    label_visibility="collapsed"
)
CURRENT_LANG = LANGUAGES[selected_language]

def translate(text):
    """Robust centralized text translation helper function."""
    if not isinstance(text, str) or CURRENT_LANG == "en" or not text.strip():
        return text
    table = UI_TRANSLATIONS.get(CURRENT_LANG, {})
    if text in table:
        return table[text]
    out = text
    # Sort keys by length descending to match full sentences before fragments
    for source, target in sorted(table.items(), key=lambda x: len(x[0]), reverse=True):
        if source in out:
            out = out.replace(source, target)
    return out

# Override Streamlit components to automatically apply translation
#
# IMPORTANT: the `streamlit` module object is cached (imported once) and
# persists across Streamlit script reruns within the same session — only
# the script's own top-level code re-executes on every rerun, not the
# `streamlit` package itself. Because of that, naively doing
# `_original_selectbox = st.selectbox` on every rerun would, after the
# first rerun, capture the *already-wrapped* function from the previous
# run instead of the real native Streamlit function, causing translation
# wrappers to stack endlessly (each rerun adding another translation
# pass on top of the last). That stacking is what corrupted widget
# return values/state and stopped the UI from cleanly reverting to
# English. To prevent this, the true native functions are captured and
# stashed on the `st` module exactly once per process; every rerun after
# that reuses the same stashed originals.
if not hasattr(st, "_trial_native_funcs"):
    st._trial_native_funcs = {
        "markdown": st.markdown,
        "caption": st.caption,
        "info": st.info,
        "warning": st.warning,
        "error": st.error,
        "success": st.success,
        "write": st.write,
        "button": st.button,
        "form_submit_button": st.form_submit_button,
        "selectbox": st.selectbox,
        "checkbox": st.checkbox,
        "radio": st.radio,
        "text_input": st.text_input,
        "number_input": st.number_input,
        "file_uploader": st.file_uploader,
        "camera_input": st.camera_input,
        "spinner": st.spinner,
    }

_native = st._trial_native_funcs
_original_markdown = _native["markdown"]
_original_caption = _native["caption"]
_original_info = _native["info"]
_original_warning = _native["warning"]
_original_error = _native["error"]
_original_success = _native["success"]
_original_write = _native["write"]
_original_button = _native["button"]
_original_form_submit_button = _native["form_submit_button"]
_original_selectbox = _native["selectbox"]
_original_checkbox = _native["checkbox"]
_original_radio = _native["radio"]
_original_text_input = _native["text_input"]
_original_number_input = _native["number_input"]
_original_file_uploader = _native["file_uploader"]
_original_camera_input = _native["camera_input"]
_original_spinner = _native["spinner"]

def _translated_markdown(body, *args, **kwargs):
    return _original_markdown(translate(body), *args, **kwargs)

def _translated_caption(body, *args, **kwargs):
    return _original_caption(translate(body), *args, **kwargs)

def _translated_info(body, *args, **kwargs):
    return _original_info(translate(body), *args, **kwargs)

def _translated_warning(body, *args, **kwargs):
    return _original_warning(translate(body), *args, **kwargs)

def _translated_error(body, *args, **kwargs):
    return _original_error(translate(body), *args, **kwargs)

def _translated_success(body, *args, **kwargs):
    return _original_success(translate(body), *args, **kwargs)

def _translated_write(*args, **kwargs):
    translated = [translate(x) if isinstance(x, str) else x for x in args]
    return _original_write(*translated, **kwargs)

def _translated_button(label, *args, **kwargs):
    return _original_button(translate(label), *args, **kwargs)

def _translated_submit(label, *args, **kwargs):
    return _original_form_submit_button(translate(label), *args, **kwargs)

def _translated_selectbox(label, options, *args, **kwargs):
    display_options = [translate(x) if isinstance(x, str) else x for x in options]
    idx = _original_selectbox(translate(label), display_options, *args, **kwargs)
    try:
        val_index = display_options.index(idx)
        return options[val_index]
    except Exception:
        return idx

def _translated_checkbox(label, *args, **kwargs):
    return _original_checkbox(translate(label), *args, **kwargs)

def _translated_radio(label, options, *args, **kwargs):
    display_options = [translate(x) if isinstance(x, str) else x for x in options]
    res = _original_radio(translate(label), display_options, *args, **kwargs)
    try:
        val_index = display_options.index(res)
        return options[val_index]
    except Exception:
        return res

def _translated_text_input(label, *args, **kwargs):
    return _original_text_input(translate(label), *args, **kwargs)

def _translated_number_input(label, *args, **kwargs):
    return _original_number_input(translate(label), *args, **kwargs)

def _translated_file_uploader(label, *args, **kwargs):
    return _original_file_uploader(translate(label), *args, **kwargs)

def _translated_camera_input(label, *args, **kwargs):
    return _original_camera_input(translate(label), *args, **kwargs)

def _translated_spinner(text="Working...", *args, **kwargs):
    return _original_spinner(translate(text), *args, **kwargs)

st.markdown = _translated_markdown
st.caption = _translated_caption
st.info = _translated_info
st.warning = _translated_warning
st.error = _translated_error
st.success = _translated_success
st.write = _translated_write
st.button = _translated_button
st.form_submit_button = _translated_submit
st.selectbox = _translated_selectbox
st.checkbox = _translated_checkbox
st.radio = _translated_radio
st.text_input = _translated_text_input
st.number_input = _translated_number_input
st.file_uploader = _translated_file_uploader
st.camera_input = _translated_camera_input
st.spinner = _translated_spinner

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
        margin-bottom: 1.2rem;
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
    .result-name { font-size: 1.5rem; font-weight: 800; margin-top: 4px; color: #f6f9f2; }
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
    div[data-testid="stFormSubmitButton"] button {
        background: #f2c744 !important;
        color: #23331b !important;
        border: none !important;
        border-radius: 30px !important;
        font-weight: 800 !important;
        padding: 0.7rem 1.6rem !important;
        font-size: 1.02rem !important;
    }
    div[data-testid="stFormSubmitButton"] button:hover { background: #f6d768 !important; }

    .info-card {
        background: rgba(255,255,255,0.06);
        border-radius: 18px;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
        border: 1px solid rgba(255,255,255,0.10);
        border-left: 5px solid #f2c744;
    }
    .info-card.mild { border-left-color: #7bd389; }
    .info-card.moderate { border-left-color: #f2c744; }
    .info-card.severe { border-left-color: #e0665a; }
    .info-card h4 {
        margin-top: 0;
        margin-bottom: 0.4rem;
        color: #f6f9f2;
    }
    .info-card p {
        margin: 0.2rem 0;
        color: #d3ddc7;
        font-size: 0.92rem;
    }
    .health-status-badge {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.92rem;
        margin-top: 0.4rem;
    }
    .health-bar-bg {
        background-color: rgba(255,255,255,0.12);
        border-radius: 10px;
        height: 14px;
        width: 100%;
        margin-top: 8px;
        overflow: hidden;
    }
    .health-bar-fill {
        height: 14px;
        border-radius: 10px;
        transition: width 0.8s cubic-bezier(0.34, 1.56, 0.64, 1);
    }

    .subsection-title {
        font-weight: 700;
        color: #f2c744;
        margin: 1.4rem 0 0.5rem 0;
        font-size: 1.05rem;
    }
    .urgency-badge {
        display: inline-block;
        padding: 0.45rem 1.1rem;
        border-radius: 20px;
        font-weight: 800;
        font-size: 0.95rem;
        margin-top: 0.3rem;
    }
    div[data-testid="stNumberInput"] label, div[data-testid="stCheckbox"] label, div[data-testid="stRadio"] label, div[data-testid="stTextInput"] label {
        color: #eef2e6 !important;
    }
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input {
        background: #ffffff !important;
        color: #1c2a17 !important;
        -webkit-text-fill-color: #1c2a17 !important;
        caret-color: #1c2a17 !important;
        font-weight: 700 !important;
        border: 1px solid rgba(255,255,255,0.25) !important;
        border-radius: 10px !important;
    }
    div[data-testid="stNumberInput"] button svg {
        fill: #1c2a17 !important;
    }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background: #ffffff !important;
        color: #1c2a17 !important;
        border-radius: 10px !important;
    }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] span {
        color: #1c2a17 !important;
    }
    ul[data-testid="stSelectboxVirtualDropdown"] li {
        color: #1c2a17 !important;
    }

    .photo-count-badge {
        display: inline-block;
        background: rgba(242,199,68,0.15);
        color: #f2c744;
        border: 1px solid rgba(242,199,68,0.4);
        border-radius: 30px;
        padding: 6px 16px;
        font-weight: 700;
        font-size: 0.9rem;
        margin-bottom: 12px;
    }
    .mini-metric-card {
        background: rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 0.9rem 0.7rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.10);
    }
    .mini-metric-num { font-size: 1.5rem; font-weight: 800; color: #f2c744; }
    .mini-metric-label { font-size: 0.75rem; color: #d3ddc7; margin-top: 2px; }
    </style>
""", unsafe_allow_html=True)

# ---------- Load model ----------
@st.cache_resource
def load_my_model():
    model = load_model("crop_model.h5")
    with open("class_names.json") as f:
        class_names = json.load(f)
    return model, class_names

try:
    model, class_names = load_my_model()
    MODEL_LOAD_ERROR = None
except Exception as e:
    model, class_names = None, []
    MODEL_LOAD_ERROR = str(e)

# ---------- Leaf Validation Gate (runs BEFORE the disease model) ----------
# The disease-classification model above can be made to assign a disease
# class to almost any image, so its own prediction/confidence must NEVER be
# used as proof that an image is a crop leaf in the first place. This section
# adds an independent "is this a supported crop leaf?" check.
#
# To plug in a dedicated ML-based leaf/non-leaf validator later, simply place
# a trained binary classifier file at LEAF_VALIDATION_MODEL_PATH below (e.g.
# "leaf_validation_model.keras", expected to output a single leaf-probability
# value). It will be picked up automatically. If that file is not present,
# the app does NOT crash — it gracefully falls back to a heuristic leaf check
# (is_supported_leaf_image / _leaf_color_ratio below) instead.
LEAF_VALIDATION_MODEL_PATH = "leaf_validation_model.keras"

@st.cache_resource
def load_leaf_validation_model():
    try:
        return load_model(LEAF_VALIDATION_MODEL_PATH)
    except Exception:
        # No dedicated leaf-validation model file found/loadable — this is
        # expected unless LEAF_VALIDATION_MODEL_PATH has been provided.
        # The app falls back to the heuristic check below.
        return None

leaf_validation_model = load_leaf_validation_model()


def _leaf_color_ratio(img):
    """
    Heuristic fallback leaf detector (used only when no dedicated
    leaf_validation_model is available). Estimates the fraction of the image
    made up of plant-leaf-like tones (greens through yellow-green and
    olive/brown, which also covers dried or diseased leaf tissue) using an
    HSV color mask. This is deliberately independent of the disease
    classification model and its output.
    """
    hsv_array = np.array(img.convert('HSV'))
    hue_deg = hsv_array[:, :, 0].astype(np.float32) * (360.0 / 255.0)
    sat = hsv_array[:, :, 1].astype(np.int32)
    val = hsv_array[:, :, 2].astype(np.int32)

    leafy_hue = (hue_deg >= 35) & (hue_deg <= 170)          # yellow-green -> green -> teal-green
    has_saturation = sat > 25                                # excludes greys: sky, walls, screenshots, skin highlights
    has_visibility = (val > 20) & (val < 250)                # excludes near-black shadows and blown-out white

    leaf_mask = leafy_hue & has_saturation & has_visibility
    return float(np.mean(leaf_mask))


def is_supported_leaf_image(img):
    """
    Independent leaf / non-leaf gate. Must be called BEFORE the disease
    model runs on an image. Returns (is_leaf: bool, error_message: str).

    Uses leaf_validation_model when available; otherwise falls back to the
    color-based heuristic. Never uses the disease-classification model's
    output to decide whether the image is a leaf.
    """
    unsupported_msg = ("⚠️ Invalid Image: This image does not appear to be a "
                        "supported crop leaf. Please upload or capture a clear "
                        "photo of a supported crop leaf.")
    try:
        if leaf_validation_model is not None:
            leaf_input = np.array(img.resize((224, 224)), dtype=np.float32) / 255.0
            leaf_input = np.expand_dims(leaf_input, axis=0)
            leaf_pred = leaf_validation_model.predict(leaf_input, verbose=0)
            leaf_prob = float(np.ravel(leaf_pred)[0])
            if leaf_prob < 0.5:
                return False, unsupported_msg
            return True, ""
        else:
            leaf_ratio = _leaf_color_ratio(img)
            if leaf_ratio < 0.18:
                return False, unsupported_msg
            return True, ""
    except Exception:
        # Fail safe: never let a validation error let an unchecked image
        # through to the disease model.
        return False, unsupported_msg

# Disease Info Database
disease_info = {
    "Pepper__bell___Bacterial_spot": {
        "name": "Bell Pepper Bacterial Spot",
        "severity": "Moderate",
        "action": "Spray copper-based bactericides early. Remove and destroy infected leaves to halt spread.",
        "precaution": "Avoid overhead irrigation as water splashes spread bacteria rapidly.",
    },
    "Pepper__bell___healthy": {
        "name": "Healthy Bell Pepper Leaf",
        "severity": "Healthy",
        "action": "No treatment required. Maintain balanced watering and optimal soil fertility.",
        "precaution": "Regularly inspect undersides of leaves for early signs of pests.",
    },
    "Potato___Early_blight": {
        "name": "Potato Early Blight",
        "severity": "Moderate",
        "action": "Apply fungicides like Mancozeb or Chlorothalonil every 7–10 days.",
        "precaution": "Practice crop rotation with non-solanaceous crops for at least 2–3 seasons.",
    },
    "Potato___Late_blight": {
        "name": "Potato Late Blight",
        "severity": "Severe",
        "action": "Apply systemic fungicides like Ridomil Gold or Cymoxanil immediately.",
        "precaution": "Destroy severely infected plants and maintain field sanitation.",
    },
    "Potato___healthy": {
        "name": "Healthy Potato Leaf",
        "severity": "Healthy",
        "action": "Crop is healthy. Ensure adequate potassium and nitrogen nutrients.",
        "precaution": "Keep foliage dry; irrigate early in the day.",
    },
    "Tomato___Bacterial_spot": {
        "name": "Tomato Bacterial Spot",
        "severity": "Moderate",
        "action": "Use copper hydroxide spray mixed with Mancozeb for better control.",
        "precaution": "Sanitize tools between handling affected plants.",
    },
    "Tomato___Early_blight": {
        "name": "Tomato Early Blight",
        "severity": "Moderate",
        "action": "Apply copper-based or chlorothalonil fungicides; prune lower infected foliage.",
        "precaution": "Mulch around soil base to prevent fungal spores from splashing up.",
    },
    "Tomato___Late_blight": {
        "name": "Tomato Late Blight",
        "severity": "Severe",
        "action": "Apply systemic fungicides (Mancozeb, Copper Oxychloride) without delay.",
        "precaution": "High humidity accelerates spread; increase plant spacing for airflow.",
    },
    "Tomato___Leaf_Mold": {
        "name": "Tomato Leaf Mold",
        "severity": "Moderate",
        "action": "Apply fungicides containing difenoconazole or copper soap.",
        "precaution": "Reduce greenhouse or crop humidity by improving air circulation.",
    },
    "Tomato___Septoria_leaf_spot": {
        "name": "Tomato Septoria Leaf Spot",
        "severity": "Moderate",
        "action": "Apply chlorothalonil or copper fungicide at the first sight of small spots.",
        "precaution": "Remove lower infected leaves to delay upward spread.",
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "name": "Tomato Two-Spotted Spider Mite",
        "severity": "Moderate",
        "action": "Apply insecticidal soap, neem oil, or specific miticides (Abamectin).",
        "precaution": "Keep fields free of weeds which harbor mites during dry periods.",
    },
    "Tomato___Target_Spot": {
        "name": "Tomato Target Spot",
        "severity": "Moderate",
        "action": "Spray fungicides like azoxystrobin or chlorothalonil.",
        "precaution": "Avoid wet leaf surfaces for extended periods.",
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "name": "Tomato Yellow Leaf Curl Virus",
        "severity": "Severe",
        "action": "Control whitefly vectors using imidacloprid or neem oil sprays. Rogue infected plants.",
        "precaution": "Use yellow sticky traps and reflective mulches to deter whiteflies.",
    },
    "Tomato___Tomato_mosaic_virus": {
        "name": "Tomato Mosaic Virus",
        "severity": "Severe",
        "action": "No chemical cure. Remove and burn infected plants immediately.",
        "precaution": "Wash hands with soap before handling healthy plants; disinfect tools.",
    },
    "Tomato___healthy": {
        "name": "Healthy Tomato Leaf",
        "severity": "Healthy",
        "action": "No treatment needed. Continue good agricultural practices.",
        "precaution": "Monitor weekly for early detection of pests.",
    },
}

# Image validation and safety check functions
def validate_prediction(img, raw_preds, class_names):
    """
    Validates image quality, resolution, sharpness, and confidence score.
    Returns (is_valid, error_key_string, confidence_percentage, predicted_class_name).
    """
    # 1. Resolution Check
    width, height = img.size
    if width < 100 or height < 100:
        return False, "Image resolution is too low. Please upload a clearer photo.", 0.0, ""

    # 2. Detail / Blur Check using Image standard deviation
    gray_img = img.convert('L')
    stat = ImageStat.Stat(gray_img)
    stddev = stat.stddev[0]
    if stddev < 15.0:  # Very blank, solid color, or extremely blurry image
        return False, "Image appears too blurry or lacks visible detail. Please provide a sharp photo.", 0.0, ""

    # 3. Model Prediction Confidence Threshold
    idx = np.argmax(raw_preds[0])
    conf = float(raw_preds[0][idx]) * 100.0
    pred_class = class_names[idx]

    if conf < CONFIDENCE_THRESHOLD:
        return False, "The model is not confident enough in this image prediction. Please ensure it is a clear leaf photo.", conf, pred_class

    return True, "", conf, pred_class

# Dynamic TTS Generator
def get_voice_audio_bytes(text_content, lang_code="en"):
    try:
        tts = gTTS(text=text_content, lang=lang_code, slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp.read()
    except Exception:
        return None

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
    st.markdown('<div class="stat-card"><div class="stat-num">4</div><div class="stat-label">' + translate("Crops covered") + '</div></div>', unsafe_allow_html=True)
with s2:
    st.markdown('<div class="stat-card"><div class="stat-num">99%+</div><div class="stat-label">' + translate("Model accuracy") + '</div></div>', unsafe_allow_html=True)
with s3:
    st.markdown('<div class="stat-card"><div class="stat-num">4</div><div class="stat-label">' + translate("Languages") + '</div></div>', unsafe_allow_html=True)
with s4:
    st.markdown('<div class="stat-card"><div class="stat-num">⚡</div><div class="stat-label">' + translate("AI Advisory") + '</div></div>', unsafe_allow_html=True)

st.write("")

# ---------- How it works ----------
st.markdown('<div class="section-header">⚡ ' + translate("How it works") + '</div>', unsafe_allow_html=True)
h1, h2, h3 = st.columns(3)
with h1:
    st.markdown('<div class="step-card"><div class="step-num">1</div><b>' + translate("Snap photos") + '</b><br><span style="opacity:0.85; font-size:0.85rem;">' + translate("Take clear photos of the affected leaves") + '</span></div>', unsafe_allow_html=True)
with h2:
    st.markdown('<div class="step-card"><div class="step-num">2</div><b>' + translate("AI analyzes") + '</b><br><span style="opacity:0.85; font-size:0.85rem;">' + translate("Model detects disease instantly") + '</span></div>', unsafe_allow_html=True)
with h3:
    st.markdown('<div class="step-card"><div class="step-num">3</div><b>' + translate("Get advice") + '</b><br><span style="opacity:0.85; font-size:0.85rem;">' + translate("Hear treatment steps in your language") + '</span></div>', unsafe_allow_html=True)

st.write("")

# ---------- Main Form & Multi-photo Input ----------
if MODEL_LOAD_ERROR:
    st.error(f"Error loading model: {MODEL_LOAD_ERROR}")
    st.stop()

st.markdown('<div class="section-header">📸 ' + translate("Scan crop leaves") + '</div>', unsafe_allow_html=True)

input_choice = st.radio(
    "Input Method",
    ["📷 Click Leaf Photo", "📁 Upload Leaf Photo"],
    horizontal=True,
    label_visibility="collapsed"
)

uploaded_files = []

if input_choice == "📷 Click Leaf Photo":
    st.markdown("##### " + translate("Capture Leaf Photo"))
    st.caption(translate("Take a clear, well-lit photo of the affected crop leaf"))
    camera_file = st.camera_input("Take photo of leaf", label_visibility="collapsed")
    if camera_file is not None:
        uploaded_files = [camera_file]
else:
    st.markdown("##### " + translate("Scan crop leaves"))
    st.caption(translate("Drag one or more leaf photos below, or click to browse"))
    file_list = st.file_uploader(
        "Upload Leaf Photo",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    if file_list:
        uploaded_files = file_list

with st.form(key="farmer_info_form"):
    st.markdown('<div class="subsection-title">📋 ' + translate("Crop Information") + '</div>', unsafe_allow_html=True)
    f1, f2 = st.columns(2)
    with f1:
        crop_name = st.selectbox(
            "Crop name",
            ["Tomato", "Potato", "Bell Pepper", "Other / Not sure"]
        )
        growth_stage = st.selectbox(
            "Crop growth stage",
            ["Seedling", "Vegetative", "Flowering", "Fruiting", "Mature"]
        )
        not_sure_age = st.checkbox("Not sure about crop age")
        if not not_sure_age:
            crop_age = st.number_input("Approximate crop age (days)", min_value=1, max_value=365, value=45)
        else:
            crop_age = None
    with f2:
        symptom_duration = st.selectbox(
            "How long have you noticed the symptoms?",
            ["Less than 1 day", "1–3 days", "4–7 days", "1–2 weeks", "More than 2 weeks"]
        )
        field_spread = st.selectbox(
            "How much of the crop appears affected?",
            ["Only one/few leaves", "Less than 25%", "25–50%", "50–75%", "More than 75%"]
        )
        recent_weather = st.selectbox(
            "Recent weather / field condition",
            ["Normal", "High rainfall", "High humidity", "Very hot", "Very dry"]
        )

    st.markdown('<div class="subsection-title">🧪 ' + translate("Have you already applied any treatment?") + '</div>', unsafe_allow_html=True)
    applied_treatment = st.radio("Have you already applied any treatment?", ["No", "Yes"], horizontal=True, label_visibility="collapsed")
    treatment_details = ""
    if applied_treatment == "Yes":
        treatment_details = st.text_input("Please specify the treatment used")

    st.markdown('<div class="subsection-title">📜 ' + translate("Has this crop shown this disease before? (optional)") + '</div>', unsafe_allow_html=True)
    prior_history = st.selectbox("Has this crop shown this disease before? (optional)", ["No", "Yes", "Not sure"], label_visibility="collapsed")
    history_count = 0
    if prior_history == "Yes":
        history_count = st.number_input("How many times has this crop shown this disease before?", min_value=1, max_value=10, value=1)

    st.markdown('<div class="subsection-title">🌱 ' + translate("Type of soil used for growing") + '</div>', unsafe_allow_html=True)
    soil_type = st.selectbox("Type of soil used for growing", ["Loamy soil", "Clayey soil", "Sandy soil", "Black soil (Regur)", "Red soil", "Alluvial soil"], label_visibility="collapsed")

    c_loc1, c_loc2 = st.columns(2)
    with c_loc1:
        village = st.text_input("Village / City (optional)")
    with c_loc2:
        district = st.text_input("District (optional)")

    submit_button = st.form_submit_button(label="🔍 " + translate("Analyze All Photos"))

if submit_button:
    if not uploaded_files:
        if input_choice == "📷 Click Leaf Photo":
            st.warning("Please capture a photo first.")
        else:
            st.warning("Please upload at least one photo first.")
    else:
        with st.spinner("Analyzing photo(s)..."):
            valid_images = []
            for file in uploaded_files:
                try:
                    bytes_data = file.read()
                    img = Image.open(io.BytesIO(bytes_data)).convert('RGB')
                    valid_images.append((file.name, img))
                except UnidentifiedImageError:
                    st.error(f"{translate('Invalid or Corrupted Image')}: {file.name}")
                except Exception as e:
                    st.error(f"{translate('Failed to load image file.')}: {file.name}")

            if not valid_images:
                st.error("No valid crop leaf images were found among the inputs provided.")
            else:
                image_results = []
                for name, img in valid_images:
                    # ---- Leaf validation gate: runs BEFORE the disease model ----
                    # Each image is validated independently, so an invalid
                    # (non-leaf) image never affects other images' results,
                    # and the disease model is never invoked for it.
                    is_leaf, leaf_err_msg = is_supported_leaf_image(img)
                    if not is_leaf:
                        image_results.append({
                            "name": name,
                            "img": img,
                            "is_valid": False,
                            "error": leaf_err_msg
                        })
                        continue

                    img_resized = img.resize((224, 224))
                    img_array = np.array(img_resized, dtype=np.float32) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)

                    preds = model.predict(img_array, verbose=0)
                    is_valid, err_msg, conf, raw_class_name = validate_prediction(img, preds, class_names)

                    if not is_valid:
                        image_results.append({
                            "name": name,
                            "img": img,
                            "is_valid": False,
                            "error": err_msg if err_msg else "⚠️ Invalid Image: This image does not appear to be a supported crop leaf. Please upload or capture a clear photo of a supported crop leaf."
                        })
                    else:
                        info = disease_info.get(raw_class_name, {
                            "name": raw_class_name.replace("_", " "),
                            "severity": "Moderate",
                            "action": "Consult agricultural expert.",
                            "precaution": "Monitor closely."
                        })
                        image_results.append({
                            "name": name,
                            "img": img,
                            "is_valid": True,
                            "class_raw": raw_class_name,
                            "disease_name": info["name"],
                            "confidence": conf,
                            "severity": info["severity"],
                            "action": info["action"],
                            "precaution": info["precaution"]
                        })

                # Separate valid leaf predictions from rejected images
                valid_preds = [r for r in image_results if r["is_valid"]]

                st.markdown('<div class="section-header">📷 ' + translate("Individual Photo Results") + '</div>', unsafe_allow_html=True)
                cols_per_row = 3
                for i in range(0, len(image_results), cols_per_row):
                    row_items = image_results[i:i+cols_per_row]
                    cols = st.columns(len(row_items))
                    for col, res in zip(cols, row_items):
                        with col:
                            st.image(res["img"], use_container_width=True)
                            if not res["is_valid"]:
                                st.error(translate(res["error"]))
                            else:
                                translated_dis = translate(res['disease_name'])
                                is_healthy = (res['severity'] == "Healthy")
                                color = "#7bd389" if is_healthy else ("#f2c744" if res['severity'] == "Moderate" else "#e0665a")
                                st.markdown(f"""
                                    <div style="background: rgba(255,255,255,0.06); padding: 12px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1); margin-top: 6px;">
                                        <div style="font-weight: 800; color: #f6f9f2; font-size: 1rem;">{translated_dis}</div>
                                        <div style="font-size: 0.82rem; color: #d3ddc7; margin-top: 4px;">{translate('Confidence')}: <b>{res['confidence']:.1f}%</b></div>
                                        <div style="font-size: 0.82rem; color: {color}; margin-top: 2px;">{translate('Severity')}: <b>{translate(res['severity'])}</b></div>
                                    </div>
                                """, unsafe_allow_html=True)

                                single_voice_text = f"{translated_dis}. {translate('Severity')}: {translate(res['severity'])}. {translate('Treatment')}: {translate(res['action'])}"
                                audio_b = get_voice_audio_bytes(single_voice_text, CURRENT_LANG)
                                if audio_b:
                                    st.audio(audio_b, format="audio/mp3")

                # If no images passed validation, block further diagnosis
                if not valid_preds:
                    st.write("")
                    st.error("❌ " + translate("None of the photos provided were valid crop leaf images suitable for disease analysis."))
                else:
                    # Multi-image Summary calculation
                    total_valid = len(valid_preds)
                    healthy_count = sum(1 for r in valid_preds if r["severity"] == "Healthy")
                    affected_count = total_valid - healthy_count
                    avg_conf = sum(r["confidence"] for r in valid_preds) / total_valid

                    disease_counts = Counter(r["disease_name"] for r in valid_preds if r["severity"] != "Healthy")
                    if disease_counts:
                        primary_disease_name = disease_counts.most_common(1)[0][0]
                        primary_info = next(r for r in valid_preds if r["disease_name"] == primary_disease_name)
                    else:
                        primary_disease_name = "Healthy"
                        primary_info = valid_preds[0]

                    st.markdown('<div class="section-header">📊 ' + translate("Overall Crop Health Assessment") + '</div>', unsafe_allow_html=True)

                    m1, m2, m3, m4, m5 = st.columns(5)
                    with m1:
                        st.markdown(f'<div class="mini-metric-card"><div class="mini-metric-num">{len(uploaded_files)}</div><div class="mini-metric-label">{translate("Photos analyzed")}</div></div>', unsafe_allow_html=True)
                    with m2:
                        st.markdown(f'<div class="mini-metric-card"><div class="mini-metric-num" style="color:#e0665a;">{affected_count}</div><div class="mini-metric-label">{translate("Affected photos")}</div></div>', unsafe_allow_html=True)
                    with m3:
                        st.markdown(f'<div class="mini-metric-card"><div class="mini-metric-num" style="color:#7bd389;">{healthy_count}</div><div class="mini-metric-label">{translate("Healthy photos")}</div></div>', unsafe_allow_html=True)
                    with m4:
                        st.markdown(f'<div class="mini-metric-card"><div class="mini-metric-num">{avg_conf:.1f}%</div><div class="mini-metric-label">{translate("Avg. confidence")}</div></div>', unsafe_allow_html=True)
                    with m5:
                        overall_risk = "Low Risk" if affected_count == 0 else ("High Risk" if any(r["severity"] == "Severe" for r in valid_preds) else "Moderate Risk")
                        risk_color = "#7bd389" if overall_risk == "Low Risk" else ("#e0665a" if overall_risk == "High Risk" else "#f2c744")
                        st.markdown(f'<div class="mini-metric-card"><div class="mini-metric-num" style="color:{risk_color};">{translate(overall_risk)}</div><div class="mini-metric-label">{translate("Overall risk")}</div></div>', unsafe_allow_html=True)

                    st.write("")

                    # Disease / Health summary card
                    is_healthy_overall = (affected_count == 0)
                    overall_title = "No disease detected" if is_healthy_overall else primary_disease_name
                    translated_overall_title = translate(overall_title)
                    dot_color = "#7bd389" if is_healthy_overall else ("#e0665a" if primary_info.get("severity") == "Severe" else "#f2c744")

                    st.markdown(f"""
                        <div class="result-card" style="border-top-color: {dot_color};">
                            <div class="result-label"><span class="status-dot" style="background-color: {dot_color};"></span>{translate("Detection result")}</div>
                            <div class="result-name">{translated_overall_title}</div>
                        </div>
                    """, unsafe_allow_html=True)

                    # Dynamic Progress & Urgency Logic
                    if symptom_duration in ["Less than 1 day", "1–3 days"]:
                        progression_text = "Early stage"
                        prog_desc = "Symptoms are recent. Early treatment yields the highest recovery rate."
                    elif symptom_duration == "4–7 days":
                        progression_text = "Progressed"
                        prog_desc = "Disease is establishing. Apply treatment promptly to prevent further spreading."
                    else:
                        progression_text = "Advanced stage"
                        prog_desc = "Disease has been active for over a week. Urgent comprehensive control is needed."

                    if is_healthy_overall:
                        urgency_text = "Preventive care"
                        urgency_color = "#7bd389"
                        urgency_desc = "Keep up good agricultural practices and standard routine inspection."
                    elif primary_info.get("severity") == "Severe" or field_spread in ["50–75%", "More than 75%"]:
                        urgency_text = "Act immediately"
                        urgency_color = "#e0665a"
                        urgency_desc = "High risk of crop damage. Apply recommended treatments within 24 hours."
                    elif primary_info.get("severity") == "Moderate":
                        urgency_text = "Act within 1–2 days"
                        urgency_color = "#f2c744"
                        urgency_desc = "Moderate threat. Plan spray application within 48 hours to manage spread."
                    else:
                        urgency_text = "Monitor closely"
                        urgency_color = "#7bd389"
                        urgency_desc = "Low immediate threat. Monitor fields and apply precautions."

                    # Detailed Recommendation Section
                    st.markdown('<div class="section-header">💡 ' + translate("Detailed Recommendation") + '</div>', unsafe_allow_html=True)
                    st.markdown(f"""
                        <div class="treatment-box">
                            <h3 style="margin-top:0; color:#f6f9f2;">🎯 {translate("Recommended Action")}</h3>
                            <p style="font-size:1.05rem; color:#eef2e6; font-weight:600;">{translate(primary_info.get("action", "Maintain regular field management."))}</p>
                            <hr style="border-color: rgba(242,199,68,0.3); margin: 12px 0;">
                            <div style="display:flex; gap:12px; align-items:center; flex-wrap:wrap;">
                                <span class="urgency-badge" style="background:{urgency_color}; color:#1c2a17;">⏱️ {translate(urgency_text)}</span>
                                <span style="font-size:0.92rem; color:#d3ddc7;">{translate(urgency_desc)}</span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                    if primary_info.get("precaution"):
                        st.warning(f"⚠️ **{translate('Important Precaution')}:** {translate(primary_info['precaution'])}")

                    # Weather Risk Context
                    if recent_weather in ["High rainfall", "High humidity"]:
                        st.info(f"🌧️ **{translate('Weather Risk')}:** {translate('High humidity or rainfall significantly speeds up fungal and bacterial disease progression. Ensure good field drainage and avoid overhead irrigation.')}")
                    elif recent_weather == "Very hot":
                        st.info(f"☀️ **{translate('Weather Risk')}:** {translate('Hot weather increases plant stress and pest population multiplication (such as spider mites). Ensure proper irrigation.')}")

                    # Farmer Info Overview
                    st.markdown('<div class="section-header">👨‍🌾 ' + translate("Farmer Information") + '</div>', unsafe_allow_html=True)
                    st.markdown(f"""
                        <div class="helpline-card">
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px;">
                                <div><b>{translate("Crop name")}:</b> {translate(crop_name)}</div>
                                <div><b>{translate("Growth Stage")}:</b> {translate(growth_stage)}</div>
                                <div><b>{translate("Soil")}:</b> {translate(soil_type)}</div>
                                <div><b>{translate("Symptoms Duration")}:</b> {translate(symptom_duration)}</div>
                                <div><b>{translate("Crop Affected")}:</b> {translate(field_spread)}</div>
                                <div><b>{translate("Location")}:</b> {village or translate('None')}, {district or translate('None')}</div>
                                <div><b>{translate("Previous Treatment")}:</b> {translate(applied_treatment)} {f'({treatment_details})' if treatment_details else ''}</div>
                                <div><b>{translate("Prior History")}:</b> {translate(prior_history)} {f'({history_count}x)' if history_count > 0 else ''}</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                    # Overall Summary Audio synthesis
                    full_summary_text = f"{translate('Overall Crop Health Assessment')}: {translated_overall_title}. {translate('Recommended Action')}: {translate(primary_info.get('action', ''))}. {translate(urgency_text)}. {translate(urgency_desc)}"
                    st.markdown('<div class="section-header">🔊 ' + translate("Voice Summary") + '</div>', unsafe_allow_html=True)
                    full_audio_bytes = get_voice_audio_bytes(full_summary_text, CURRENT_LANG)
                    if full_audio_bytes:
                        st.audio(full_audio_bytes, format="audio/mp3")

# ---------- Helpline & Roadmap Footer ----------
st.markdown('<div class="section-header">📞 ' + translate("Farmer helpline & support") + '</div>', unsafe_allow_html=True)
h_col1, h_col2 = st.columns(2)
with h_col1:
    st.markdown(f"""
        <div class="helpline-card">
            <b style="color:#f2c744; font-size:1.05rem;">{translate('Kisan Call Centre (Toll-Free): 1800-180-1551')}</b><br>
            <span style="font-size:0.9rem; color:#d3ddc7;">{translate('Krishi Vigyan Kendra (KVK) Network')} · {translate('Department of Agriculture, Maharashtra')}</span>
        </div>
    """, unsafe_allow_html=True)

with h_col2:
    st.markdown(f"""
        <div class="helpline-card">
            <b style="color:#f2c744; font-size:1.05rem;">🌾 {translate('Expanding crop coverage')}</b><br>
            <span style="font-size:0.88rem; color:#d3ddc7;">{translate('Current model covers Tomato, Potato, and Bell Pepper leaf diseases.')}<br>{translate('Expanding to Sugarcane, Cotton, Soybean, and Rice in upcoming versions.')}</span>
        </div>
    """, unsafe_allow_html=True)

st.markdown(f"""
    <div class="footer-note">
        {translate('AI predictions may vary based on photo quality. Always consult a local agricultural officer or expert for major crop decisions.')}<br>
        <b>KrishiRakshak AI</b> · SIH 2026 · Smart India Hackathon
    </div>
""", unsafe_allow_html=True)
