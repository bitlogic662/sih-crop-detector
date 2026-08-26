import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import json

# Load model and class names
model = load_model("crop_model.h5")
with open("class_names.json") as f:
    class_names = json.load(f)

st.set_page_config(page_title="Crop Disease Detector", page_icon="🌾")
st.title("🌾 Early Detection of Crop Diseases & Pest Infestations")
st.write("Upload a photo of a crop leaf to detect disease and get treatment advice.")

uploaded = st.file_uploader("Upload a leaf/crop photo", type=["jpg", "png", "jpeg"])

treatments = {
    "Pepper__bell___Bacterial_spot": "Apply copper-based bactericide. Avoid overhead watering and remove infected leaves.",
    "Potato___Early_blight": "Apply fungicide (Chlorothalonil or Mancozeb). Rotate crops and remove infected debris.",
    "Tomato_Late_blight": "Apply copper-based fungicide immediately. Remove and destroy infected plants to prevent spread.",
    "Tomato_healthy": "No disease detected. Continue regular monitoring and good field hygiene."
}

if uploaded:
    img = Image.open(uploaded).convert("RGB").resize((224, 224))
    st.image(img, caption="Uploaded Image", use_container_width=True)

    arr = np.expand_dims(np.array(img) / 255.0, axis=0)
    pred = model.predict(arr)
    result = class_names[np.argmax(pred)]
    confidence = float(np.max(pred)) * 100

    st.subheader(f"🔍 Detected: {result.replace('_', ' ')}")
    st.write(f"Confidence: {confidence:.1f}%")

    st.info(f"**Recommended action:** {treatments.get(result, 'Consult a local agriculture expert.')}")

    # Voice output
    from gtts import gTTS
    lang_choice = st.selectbox("Voice output language", ["English", "Hindi", "Marathi"])
    lang_map = {"English": "en", "Hindi": "hi", "Marathi": "mr"}

    if st.button("🔊 Play voice advice"):
        tts = gTTS(f"{result.replace('_', ' ')} detected. {treatments.get(result, '')}", lang=lang_map[lang_choice])
        tts.save("output.mp3")
        st.audio("output.mp3")