"""
KrishiRakshak AI - Crop Disease Detection System
==================================================
Existing app structure preserved (model loading, prediction logic, UI theme).
4 NEW hackathon features added — each new block is marked with:
    # === NEW FEATURE: ... ===
so you can see exactly what was added vs. what already existed.
"""

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# ======================================================
# EXISTING CODE — PAGE CONFIG & THEME
# ======================================================
st.set_page_config(
    page_title="KrishiRakshak AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Existing dark/green agricultural theme (kept as-is, only extended with
# a few extra CSS classes for the new info cards — see NEW FEATURE CSS below)
CUSTOM_CSS = """
<style>
    .stApp {
        background-color: #0e1a13;
        color: #e8f5e9;
    }
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #4caf50;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        color: #a5d6a7;
        margin-bottom: 1.5rem;
        font-size: 1rem;
    }
    .result-card {
        background-color: #16241a;
        border: 1px solid #2e7d32;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .result-card h3 {
        color: #81c784;
        margin-top: 0;
    }
    div[data-testid="stMetric"] {
        background-color: #16241a;
        border: 1px solid #2e7d32;
        border-radius: 10px;
        padding: 0.6rem;
    }
    /* === NEW FEATURE CSS: info cards for duration / severity / dashboard === */
    .info-card {
        background-color: #14231a;
        border-left: 5px solid #4caf50;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.9rem;
    }
    .info-card.mild { border-left-color: #8bc34a; }
    .info-card.moderate { border-left-color: #ffb300; }
    .info-card.severe { border-left-color: #e53935; }
    .info-card h4 {
        margin-top: 0;
        margin-bottom: 0.4rem;
        color: #c8e6c9;
    }
    .info-card p {
        margin: 0.15rem 0;
        color: #dcedc8;
        font-size: 0.95rem;
    }
    .health-status-badge {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
    }
    @media (max-width: 600px) {
        .main-title { font-size: 1.7rem; }
        .result-card, .info-card { padding: 0.9rem 1rem; }
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ======================================================
# EXISTING CODE — MODEL / CLASS CONFIG
# ======================================================
# TODO: replace with your actual trained model path
MODEL_PATH = "model/krishirakshak_model.h5"

# TODO: replace this with your EXACT class_names list, in the SAME order
# your model was trained/exported with. This list is only a placeholder
# example (PlantVillage-style naming) so the app is runnable end-to-end.
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___healthy",
]

IMG_SIZE = (224, 224)  # TODO: match your model's expected input size


@st.cache_resource
def load_model():
    """Existing model loader (cached so it only loads once)."""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


def preprocess_image(image: Image.Image):
    """Existing preprocessing logic: resize + normalize + add batch dim."""
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    arr = np.array(image).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict(model, image: Image.Image):
    """Existing prediction logic. Returns (predicted_class, confidence, raw_preds)."""
    processed = preprocess_image(image)
    preds = model.predict(processed, verbose=0)[0]
    predicted_idx = int(np.argmax(preds))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(preds[predicted_idx]) * 100.0
    return predicted_class, confidence, preds


def format_label(raw_class_name: str):
    """Existing helper: turns 'Tomato___Early_blight' into ('Tomato', 'Early blight')."""
    parts = raw_class_name.split("___")
    crop = parts[0].replace("_", " ")
    disease = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
    return crop, disease


# ======================================================
# === NEW FEATURE 4 (part A): BASIC IMAGE VALIDATION ===
# ======================================================
def validate_prediction(image: Image.Image, confidence: float, disease_label: str,
                         confidence_threshold: float = 60.0):
    """
    Lightweight, dependency-free sanity checks so obviously unsuitable
    images aren't confidently presented as a disease diagnosis.

    This performs:
      1. A basic image-quality check (size too small / near-blank image).
      2. A rough "does this look plant-like" heuristic based on green-channel
         dominance (crop leaves are usually green-dominant; this is NOT a
         scientific classifier, just a cheap sanity filter).
      3. A confidence-threshold check.

    Returns a dict:
      {
        "is_valid": bool,
        "warnings": [list of warning strings to show the user]
      }
    """
    warnings = []
    is_valid = True

    # 1. Basic resolution check
    width, height = image.size
    if width < 50 or height < 50:
        warnings.append("⚠️ The uploaded image resolution is very low. Please upload a clearer photo.")
        is_valid = False

    # 2. Rough "plant-like" heuristic (green dominance check)
    try:
        small = image.convert("RGB").resize((64, 64))
        arr = np.array(small).astype("float32")
        mean_r = arr[:, :, 0].mean()
        mean_g = arr[:, :, 1].mean()
        mean_b = arr[:, :, 2].mean()
        # Near-blank / uniform image (e.g. blank wall, plain background)
        std_total = arr.std()
        if std_total < 8:
            warnings.append("⚠️ The image appears to have very little detail. "
                             "Please upload a focused photo of the affected leaf.")
            is_valid = False
        # If green is not at least reasonably prominent, the photo may not be a leaf.
        # This is a soft heuristic only — it does not block prediction outright,
        # it just adds a caution note.
        if mean_g < mean_r * 0.85 and mean_g < mean_b * 0.85:
            warnings.append("ℹ️ This image doesn't show strong green/leaf-like coloring. "
                             "Please make sure the photo is a close-up of a crop leaf for best results.")
    except Exception:
        # If the heuristic itself fails for any reason, don't block the app —
        # just skip this soft check.
        pass

    # 3. Confidence threshold check
    if confidence < confidence_threshold:
        warnings.append("⚠️ Low confidence prediction. Please upload a clearer image of the affected crop leaf.")
        is_valid = False

    return {"is_valid": is_valid, "warnings": warnings}


# ======================================================
# === NEW FEATURE 2: DISEASE SEVERITY ASSESSMENT ===
# ======================================================
# Optional disease-specific severity hints. If a disease isn't listed here,
# severity falls back to confidence-based rules only. Keys should match the
# "disease" part of your class name (after replacing underscores with spaces).
DISEASE_SEVERITY_HINTS = {
    "late blight": "aggressive",   # tends to progress fast
    "black rot": "aggressive",
    "bacterial spot": "moderate_prone",
}


def get_severity(disease: str, confidence: float):
    """
    Estimates an AI-based severity level using model confidence combined
    with optional disease-specific hints.

    IMPORTANT: This is an ESTIMATED, AI-assisted indicator only —
    it is explicitly NOT a scientifically validated severity measurement.

    Returns a dict: {level, explanation, action, css_class}
    """
    disease_key = disease.strip().lower()
    hint = DISEASE_SEVERITY_HINTS.get(disease_key)

    # Base thresholds from confidence
    if confidence >= 85:
        level = "Severe"
    elif confidence >= 65:
        level = "Moderate"
    else:
        level = "Mild"

    # Nudge severity up slightly for diseases known to progress aggressively,
    # since even a moderate-confidence detection of these is worth extra caution.
    if hint == "aggressive" and level == "Mild":
        level = "Moderate"

    severity_info = {
        "Mild": {
            "explanation": "The AI model detects early or low-intensity visual symptoms.",
            "action": "Monitor the crop regularly and remove visibly affected leaves if appropriate.",
            "css_class": "mild",
        },
        "Moderate": {
            "explanation": "The AI model detects clearer, more established disease symptoms.",
            "action": "Increase monitoring frequency and consider suitable crop-protection measures.",
            "css_class": "moderate",
        },
        "Severe": {
            "explanation": "The AI model detects strong, widespread visual symptoms of disease.",
            "action": "Consult a local agricultural expert or agriculture officer and take prompt action.",
            "css_class": "severe",
        },
    }

    result = severity_info[level]
    return {
        "level": level,
        "explanation": result["explanation"],
        "action": result["action"],
        "css_class": result["css_class"],
    }


# ======================================================
# === NEW FEATURE 1: DISEASE DURATION / PROGRESSION ===
# ======================================================
def get_disease_duration(severity_level: str):
    """
    Returns an approximate, general progression estimate tied to severity.
    These are general educational ranges, NOT precise agronomic predictions —
    actual progression depends heavily on crop type, weather, and field
    conditions, and this is clearly communicated to the user in the UI.
    """
    duration_map = {
        "Mild": "Early stage: approx. 1–3 days since symptoms likely began",
        "Moderate": "Moderate stage: approx. 3–7 days of symptom development",
        "Severe": "Advanced stage: approx. 7+ days of symptom development",
    }
    return duration_map.get(severity_level, "Progression estimate unavailable")


# ======================================================
# === NEW FEATURE 3: CROP HEALTH PROGRESS DASHBOARD ===
# ======================================================
def get_crop_health_progress(is_healthy: bool, severity_level: str):
    """
    Converts detection result into a simple visual health percentage
    and a short risk-status label. This is a VISUAL indicator only,
    not a scientific health measurement.

    Returns dict: {percent, status_label, status_color}
    """
    if is_healthy:
        return {"percent": 100, "status_label": "Healthy", "status_color": "#4caf50"}

    mapping = {
        "Mild": {"percent": 75, "status_label": "Low Risk", "status_color": "#8bc34a"},
        "Moderate": {"percent": 50, "status_label": "Moderate Risk", "status_color": "#ffb300"},
        "Severe": {"percent": 25, "status_label": "High Risk", "status_color": "#e53935"},
    }
    return mapping.get(severity_level, {"percent": 50, "status_label": "Moderate Risk", "status_color": "#ffb300"})


# ======================================================
# EXISTING CODE — SIDEBAR
# ======================================================
with st.sidebar:
    st.markdown("### 🌾 KrishiRakshak AI")
    st.markdown("AI-based crop disease detection to help farmers identify "
                "crop diseases early and take timely action.")
    st.markdown("---")
    st.markdown("**Supported crops (example):**")
    st.markdown("- Apple\n- Corn\n- Potato\n- Tomato")
    st.markdown("---")
    st.caption("⚠️ AI predictions are advisory and should be verified "
                "with an agricultural expert when necessary.")

# ======================================================
# EXISTING CODE — MAIN UI HEADER
# ======================================================
st.markdown('<div class="main-title">🌾 KrishiRakshak AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload a crop leaf image to detect disease using AI</div>',
            unsafe_allow_html=True)

model = load_model()

if model is None:
    st.warning(
        "⚠️ Model file not found at `{}`. Place your trained `.h5`/`.keras` model "
        "at that path (or update `MODEL_PATH`) to enable live predictions. "
        "The rest of the interface below still renders normally.".format(MODEL_PATH)
    )

uploaded_file = st.file_uploader(
    "Upload a crop leaf image (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"],
)

# ======================================================
# EXISTING CODE — PREDICTION FLOW (extended with new features below)
# ======================================================
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
    except Exception:
        st.error("⚠️ Could not read this file as an image. Please upload a valid JPG/PNG image.")
        image = None

    if image is not None:
        col_img, col_result = st.columns([1, 1.3])

        with col_img:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        with col_result:
            if model is not None:
                with st.spinner("Analyzing image..."):
                    predicted_class, confidence, raw_preds = predict(model, image)
                crop, disease = format_label(predicted_class)
                is_healthy = "healthy" in disease.lower()

                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown("### 🔍 Detection Result")
                c1, c2 = st.columns(2)
                c1.metric("Detected Crop", crop)
                c2.metric("Condition", "Healthy ✅" if is_healthy else disease.title())
                st.markdown('</div>', unsafe_allow_html=True)

                # === NEW FEATURE 4 (part B): CONFIDENCE DISPLAY + VALIDATION ===
                validation = validate_prediction(image, confidence, disease)

                st.markdown("#### 🎯 AI Confidence")
                st.metric("Model Confidence", f"{confidence:.1f}%")
                st.progress(min(max(confidence / 100.0, 0.0), 1.0))

                for warning_msg in validation["warnings"]:
                    if "Low confidence" in warning_msg or "resolution" in warning_msg or "little detail" in warning_msg:
                        st.warning(warning_msg)
                    else:
                        st.info(warning_msg)

                st.caption("Note: AI predictions are advisory and should be verified with an "
                           "agricultural expert when necessary. This tool does not claim 100% accuracy.")

                # Only show disease-specific analysis if the model is reasonably
                # confident AND the leaf isn't predicted healthy.
                if not is_healthy and validation["is_valid"]:

                    # === NEW FEATURE 2: SEVERITY CARD ===
                    severity = get_severity(disease, confidence)
                    st.markdown(
                        f'<div class="info-card {severity["css_class"]}">'
                        f'<h4>🩺 Estimated Disease Severity: {severity["level"]}</h4>'
                        f'<p>{severity["explanation"]}</p>'
                        f'<p><b>Recommended action:</b> {severity["action"]}</p>'
                        f'<p style="font-size:0.8rem; color:#a5d6a7;">'
                        f'This is an AI-based/estimated severity assessment, not a scientifically '
                        f'validated diagnosis.</p>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    # === NEW FEATURE 1: DISEASE DURATION / PROGRESSION CARD ===
                    duration_text = get_disease_duration(severity["level"])
                    st.markdown(
                        f'<div class="info-card">'
                        f'<h4>⏳ Estimated Disease Progression</h4>'
                        f'<p>{duration_text}</p>'
                        f'<p style="font-size:0.8rem; color:#a5d6a7;">'
                        f'These are approximate estimates. Actual progression depends on crop type, '
                        f'weather conditions, and field/infection conditions.</p>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                elif is_healthy:
                    st.success("✅ No disease detected. Your crop appears healthy!")

        # ======================================================
        # === NEW FEATURE 3: CROP HEALTH PROGRESS DASHBOARD ===
        # ======================================================
        if model is not None and image is not None:
            st.markdown("---")
            st.markdown("## 📊 Crop Health Progress")

            if is_healthy:
                health = get_crop_health_progress(is_healthy=True, severity_level="Healthy")
                display_severity = "Healthy"
                display_duration = "No active disease detected"
            else:
                severity = get_severity(disease, confidence)
                health = get_crop_health_progress(is_healthy=False, severity_level=severity["level"])
                display_severity = severity["level"]
                display_duration = get_disease_duration(severity["level"])

            d1, d2, d3 = st.columns(3)
            d1.metric("Crop", crop)
            d2.metric("Disease", "None" if is_healthy else disease.title())
            d3.metric("Severity", display_severity)

            e1, e2 = st.columns(2)
            e1.metric("AI Confidence", f"{confidence:.1f}%")
            e2.metric("Progression Estimate", display_duration if is_healthy else display_duration.split(":")[0])

            st.markdown(f"**Overall Crop Health: {health['percent']}%**")
            st.progress(health["percent"] / 100.0)

            st.markdown(
                f'<span class="health-status-badge" '
                f'style="background-color:{health["status_color"]}22; '
                f'color:{health["status_color"]}; border:1px solid {health["status_color"]};">'
                f'Your crop health status: {health["status_label"]}'
                f'</span>',
                unsafe_allow_html=True,
            )
            st.caption("This progress indicator is a simplified visual guide, not a scientific "
                       "measurement of plant health.")
else:
    st.info("👆 Upload a crop leaf image above to get started.")

# ======================================================
# EXISTING CODE — FOOTER
# ======================================================
st.markdown("---")
st.caption("🌾 KrishiRakshak AI — Built to support farmers with early, AI-assisted crop disease insights. "
           "Always verify critical decisions with a local agricultural expert.")
