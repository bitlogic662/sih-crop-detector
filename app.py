"""
KrishiRakshak AI — Streamlit App
Merges the dashboard UI (HTML/CSS design ported natively into Streamlit)
with real MobileNetV2 model inference and a popup field-questionnaire.
"""

import json
from datetime import datetime

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="KrishiRakshak AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# GLOBAL STYLE (ported from style.css — dark green theme)
# =========================================================

st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}

    .stApp {
        background:
            radial-gradient(circle at 80% 10%, rgba(57, 150, 93, 0.15), transparent 30%),
            linear-gradient(135deg, #071b13, #0b2a1d 45%, #071812);
        color: #edf7f0;
    }

    section[data-testid="stSidebar"] {
        background: rgba(4, 20, 14, 0.95);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] * {
        color: #edf7f0 !important;
    }

    div.stButton > button {
        background: linear-gradient(135deg, #38a861, #54cf7d);
        color: #06150d;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        padding: 0.6rem 1.1rem;
        transition: 0.25s;
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(63, 199, 111, .25);
        color: #06150d;
    }

    .krk-card {
        background: rgba(12, 38, 26, .72);
        border: 1px solid rgba(255,255,255,.07);
        border-radius: 18px;
        padding: 22px;
        margin-bottom: 16px;
    }

    .krk-badge {
        display: inline-block;
        background: rgba(75, 170, 103, .12);
        border: 1px solid rgba(90, 180, 116, .25);
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 11px;
        color: #9be0af;
        margin-bottom: 10px;
    }

    .krk-small-label {
        color: #64d88a;
        font-size: 10px;
        letter-spacing: 1.5px;
        font-weight: bold;
    }

    .krk-severity-severe {
        background: rgba(225, 75, 65, .15);
        color: #ff7770;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: bold;
    }

    .krk-treatment-box {
        margin-top: 18px;
        padding: 20px;
        border-radius: 15px;
        background: linear-gradient(135deg, rgba(30, 92, 53, .5), rgba(10, 40, 25, .8));
        border: 1px solid rgba(91, 190, 117, .2);
    }

    .krk-advice-item {
        display: flex;
        gap: 12px;
        padding: 10px 0;
        border-bottom: 1px solid rgba(255,255,255,.06);
    }

    .krk-advice-item strong {
        display: block;
        margin-bottom: 4px;
        font-size: 13px;
        color: #edf7f0;
    }

    .krk-advice-item p {
        color: #a4b8aa;
        font-size: 12px;
        line-height: 1.55;
        margin: 0;
    }

    .krk-advice-icon {
        font-size: 17px;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# MODEL LOADING
# =========================================================

MODEL_PATH = "crop_model.h5"
CLASS_NAMES_PATH = "class_names.json"
IMG_SIZE = (224, 224)  # standard MobileNetV2 input — verify against your training script


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


@st.cache_resource
def load_class_names():
    with open(CLASS_NAMES_PATH, "r") as f:
        return json.load(f)


def predict_disease(image: Image.Image):
    """
    Runs the uploaded image through the MobileNetV2 model.

    NOTE: This uses tf.keras.applications.mobilenet_v2.preprocess_input,
    the standard preprocessing for MobileNetV2 transfer learning
    (scales pixels to [-1, 1]). If your training pipeline instead used
    a plain rescale of 1./255, swap the preprocessing line below —
    otherwise predictions will be systematically wrong even though the
    model itself is accurate.
    """
    model = load_model()
    class_names = load_class_names()

    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32")
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    confidence = float(preds[top_idx]) * 100

    disease_raw = class_names[top_idx]
    disease_display = disease_raw.replace("___", " ").replace("__", " ").replace("_", " ").strip()

    return disease_raw, disease_display, confidence


# =========================================================
# DISPLAY-NAME MAPPING (raw class -> friendly label used in advice logic)
# =========================================================

DISEASE_DISPLAY_MAP = {
    "Pepper__bell___Bacterial_spot": "Bell Pepper Bacterial Spot",
    "Potato___Early_blight": "Potato Early Blight",
    "Tomato_Late_blight": "Tomato Late Blight",
    "Tomato_healthy": "Healthy",
}


# =========================================================
# QUESTIONNAIRE DATA
# =========================================================

QUESTIONS = [
    {
        "key": "spread",
        "title": "How quickly are the symptoms spreading?",
        "text": "This helps estimate the urgency of the situation.",
        "options": ["Slowly", "Rapidly", "Not sure"],
    },
    {
        "key": "sprayed",
        "title": "Have you sprayed any pesticide or fungicide recently?",
        "text": "Knowing about previous sprays helps prevent unnecessary repeat applications.",
        "options": ["Yes", "No", "Not sure"],
    },
    {
        "key": "sprayType",
        "title": "What type of product was sprayed?",
        "text": "If you are unsure about the product, select \"Don't know\".",
        "options": ["Fungicide", "Insecticide", "Both", "Don't know", "Not applicable"],
    },
    {
        "key": "weather",
        "title": "What has the recent weather been like?",
        "text": "Rainfall and humidity can influence disease development.",
        "options": ["Mostly dry", "Some rain", "Heavy rain / high humidity", "Not sure"],
    },
    {
        "key": "location",
        "title": "Where did you first notice the symptoms?",
        "text": "The location of symptoms provides additional field context.",
        "options": ["Lower leaves", "New leaves", "Fruit or stem", "Whole plant", "Not sure"],
    },
]


# =========================================================
# TREATMENT ADVICE LOGIC (ported from generateTreatmentAdvice in script.js)
# =========================================================

def generate_treatment_advice(disease_display, field_answers):
    advice = []

    if disease_display == "Tomato Late Blight":
        advice.append({
            "icon": "🚨", "title": "Immediate Action",
            "text": "Inspect the surrounding tomato plants because late blight can spread quickly "
                    "under favorable conditions. Separate or manage severely affected plant material "
                    "where appropriate.",
        })
        advice.append({
            "icon": "🌿", "title": "Field Management",
            "text": "Improve air circulation and avoid unnecessary prolonged leaf wetness. "
                    "Continue regular scouting of nearby plants.",
        })
        advice.append({
            "icon": "💊", "title": "Treatment",
            "text": "Use only a locally approved late-blight management product when recommended "
                    "by the agricultural extension service or product label. Follow the label for "
                    "dose, application interval and pre-harvest requirements.",
        })
        advice.append({
            "icon": "🛡️", "title": "Integrated Pest Management",
            "text": "Combine field sanitation, resistant or tolerant varieties where available, "
                    "weather-based monitoring and appropriate chemical control rather than relying "
                    "only on repeated spraying.",
        })

    elif disease_display == "Potato Early Blight":
        advice.append({
            "icon": "🚨", "title": "Immediate Action",
            "text": "Inspect nearby potato plants for additional lesions and monitor whether the "
                    "affected area is increasing.",
        })
        advice.append({
            "icon": "🧹", "title": "Field Sanitation",
            "text": "Remove severely affected plant material where appropriate and maintain good "
                    "field hygiene.",
        })
        advice.append({
            "icon": "💊", "title": "Treatment",
            "text": "Use a locally approved early-blight management product only according to the "
                    "product label or agricultural extension recommendation.",
        })
        advice.append({
            "icon": "🌧️", "title": "Moisture Management",
            "text": "Avoid prolonged leaf wetness and improve field conditions that allow foliage "
                    "to dry.",
        })

    elif disease_display == "Bell Pepper Bacterial Spot":
        advice.append({
            "icon": "🔬", "title": "Confirm the Diagnosis",
            "text": "Because bacterial and fungal leaf symptoms can sometimes look similar, "
                    "consider expert or laboratory confirmation when the diagnosis is uncertain.",
        })
        advice.append({
            "icon": "🧹", "title": "Sanitation",
            "text": "Remove severely affected plant material where appropriate and avoid spreading "
                    "contaminated plant debris between fields.",
        })
        advice.append({
            "icon": "💧", "title": "Water Management",
            "text": "Avoid unnecessary overhead irrigation and reduce prolonged leaf wetness where "
                    "practical.",
        })
        advice.append({
            "icon": "🧑‍🌾", "title": "Treatment",
            "text": "Follow locally approved bacterial-disease management recommendations. Do not "
                    "automatically repeat a chemical treatment without checking the label and "
                    "expert guidance.",
        })

    elif disease_display == "Healthy":
        advice.append({
            "icon": "✅", "title": "No Disease Detected",
            "text": "The plant appears healthy. Continue regular monitoring to catch any early "
                    "signs of disease.",
        })
        advice.append({
            "icon": "🔎", "title": "Preventive Scouting",
            "text": "Keep scouting nearby plants on a regular schedule, especially after rain or "
                    "high-humidity periods.",
        })

    else:
        advice.append({
            "icon": "🔎", "title": "Monitor the Crop",
            "text": "Continue regular field scouting and monitor whether symptoms increase.",
        })
        advice.append({
            "icon": "🧑‍🌾", "title": "Expert Validation",
            "text": "Seek agricultural extension assistance if symptoms become severe or the "
                    "diagnosis is uncertain.",
        })

    # Personalize using field answers
    if field_answers.get("spread") == "Rapidly":
        advice.insert(0, {
            "icon": "⚠️", "title": "Urgent: Rapid Spread Reported",
            "text": "Because you reported rapid symptom spread, inspect the surrounding field as "
                    "soon as possible and consider expert validation.",
        })

    if field_answers.get("sprayed") == "Yes":
        advice.append({
            "icon": "⚠️", "title": "Previous Spray Detected",
            "text": "You reported a recent spray. Do not automatically repeat the same product. "
                    "Check the product's active ingredient, label interval and pre-harvest "
                    "requirements before another application.",
        })

    if field_answers.get("sprayed") == "Not sure":
        advice.append({
            "icon": "📋", "title": "Check Previous Spray Records",
            "text": "If possible, check the previous product name, active ingredient and "
                    "application date before deciding on another treatment.",
        })

    if field_answers.get("weather") == "Heavy rain / high humidity":
        advice.append({
            "icon": "🌧️", "title": "Weather Alert",
            "text": "You reported heavy rain or high humidity. Increase field scouting because "
                    "wet conditions can favor development of several crop diseases.",
        })
    elif field_answers.get("weather") == "Some rain":
        advice.append({
            "icon": "🌦️", "title": "Weather Monitoring",
            "text": "Recent rainfall was reported. Continue monitoring the crop, especially if "
                    "humidity remains high.",
        })

    if field_answers.get("location") == "Lower leaves":
        advice.append({
            "icon": "🍃", "title": "Symptom Location",
            "text": "You first noticed symptoms on lower leaves. Continue checking whether "
                    "lesions are moving toward newer foliage.",
        })

    if field_answers.get("location") == "Whole plant":
        advice.append({
            "icon": "🚨", "title": "Widespread Symptoms",
            "text": "Symptoms were reported across the whole plant. Prioritize field inspection "
                    "and expert validation.",
        })

    advice.append({
        "icon": "🛡️", "title": "Safe Use Reminder",
        "text": "Always follow the approved product label and local agricultural recommendations. "
                "Wear the required protective equipment and observe the specified waiting period "
                "before harvest.",
    })

    return advice


# =========================================================
# SESSION STATE INIT
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"
if "diag_step" not in st.session_state:
    st.session_state.diag_step = 0
if "diag_answers" not in st.session_state:
    st.session_state.diag_answers = {}
if "diag_result" not in st.session_state:
    st.session_state.diag_result = None
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "show_questionnaire" not in st.session_state:
    st.session_state.show_questionnaire = False


# =========================================================
# QUESTIONNAIRE POPUP (real modal via st.dialog)
# =========================================================

@st.dialog("Field Context")
def questionnaire_popup():
    step = st.session_state.diag_step
    q = QUESTIONS[step]

    st.progress((step + 1) / len(QUESTIONS))
    st.caption(f"Question {step + 1} of {len(QUESTIONS)}")

    st.markdown(f"### {q['title']}")
    st.write(q["text"])

    answer = st.radio(
        "Select an option",
        q["options"],
        key=f"diag_q_{step}",
        label_visibility="collapsed",
        index=None,
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        if step > 0:
            if st.button("← Back", use_container_width=True):
                st.session_state.diag_step -= 1
                st.rerun()
        else:
            if st.button("✕ Cancel", use_container_width=True):
                st.session_state.show_questionnaire = False
                st.session_state.diag_step = 0
                st.session_state.diag_answers = {}
                st.rerun()

    with col2:
        is_last = step == len(QUESTIONS) - 1
        label = "Finish ✓" if is_last else "Next →"
        if st.button(label, use_container_width=True, disabled=answer is None):
            st.session_state.diag_answers[q["key"]] = answer

            if is_last:
                # Run inference now that all answers are collected
                image = st.session_state.uploaded_image
                disease_raw, disease_display, confidence = predict_disease(image)
                st.session_state.diag_result = {
                    "disease_raw": disease_raw,
                    "disease_display": disease_display,
                    "confidence": confidence,
                    "answers": dict(st.session_state.diag_answers),
                }
                st.session_state.diag_step = 0
                st.session_state.diag_answers = {}
                st.session_state.show_questionnaire = False
                st.rerun()
            else:
                st.session_state.diag_step += 1
                st.rerun()


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:
    st.markdown("## 🌾 KrishiRakshak")
    st.caption("AI Crop Intelligence")
    st.markdown("---")

    pages = ["Dashboard", "AI Diagnosis", "Risk Forecast", "Hotspot Map", "Field Monitoring", "Advisories"]
    icons = ["🏠", "🔬", "🌦️", "📍", "🌱", "💡"]

    for page, icon in zip(pages, icons):
        if st.button(f"{icon} {page}", use_container_width=True,
                     type="primary" if st.session_state.page == page else "secondary"):
            st.session_state.page = page
            st.rerun()

    st.markdown("---")
    st.markdown("🟢 **Offline Ready**")
    st.caption("SIH 2026 · SIH26131")


# =========================================================
# TOPBAR
# =========================================================

top_left, top_right = st.columns([3, 1])
with top_left:
    st.markdown("### Crop Health Intelligence")
    st.caption(datetime.now().strftime("%A, %d %B %Y"))
with top_right:
    st.selectbox("Language", ["English", "Hindi", "Marathi", "Kannada"], label_visibility="collapsed")

st.markdown("---")


# =========================================================
# DASHBOARD PAGE
# =========================================================

if st.session_state.page == "Dashboard":

    st.markdown('<div class="krk-badge">🏛️ Government of Maharashtra · SIH 2026</div>', unsafe_allow_html=True)
    st.markdown("# Early Crop Disease Intelligence")
    st.write("Detect crop diseases early, understand the risk, and receive personalized management advice.")

    if st.button("🔬 Start AI Diagnosis"):
        st.session_state.page = "AI Diagnosis"
        st.rerun()

    st.markdown("")
    c1, c2, c3, c4 = st.columns(4)
    for col, icon, val, label, sub in [
        (c1, "🌾", "1,248", "Fields Monitored", "↑ 12% this month"),
        (c2, "🚨", "37", "Active Alerts", "Needs attention"),
        (c3, "🔬", "326", "Cases Detected", "↑ 8% this month"),
        (c4, "⚡", "2.4h", "Average Response", "↓ 18% faster"),
    ]:
        with col:
            st.markdown(f"""
            <div class="krk-card">
                <div style="font-size:23px;">{icon}</div>
                <h2>{val}</h2>
                <p style="color:#91aa9d;font-size:13px;">{label}</p>
                <span style="color:#64d88a;font-size:11px;">{sub}</span>
            </div>
            """, unsafe_allow_html=True)

    col_a, col_b = st.columns([1.4, 1])
    with col_a:
        st.markdown("""
        <div class="krk-card">
            <h3>🌦️ Current Field Weather</h3>
            <h1 style="margin:10px 0;">27°C</h1>
            <p>Partly Cloudy</p>
            <p style="margin-top:15px;">💧 78% Humidity &nbsp;|&nbsp; 🌧️ 62% Rain Risk &nbsp;|&nbsp; 💨 11 km/h Wind</p>
            <p style="margin-top:10px;">Late Blight Risk: <b>72%</b></p>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown('<div class="krk-card"><h3>⚡ Quick Actions</h3></div>', unsafe_allow_html=True)
        if st.button("📷 Analyze Crop Photo", use_container_width=True):
            st.session_state.page = "AI Diagnosis"
            st.rerun()
        if st.button("🌦️ View Disease Risk", use_container_width=True):
            st.session_state.page = "Risk Forecast"
            st.rerun()
        if st.button("📍 Check Hotspots", use_container_width=True):
            st.session_state.page = "Hotspot Map"
            st.rerun()
        if st.button("💡 View Treatment Advice", use_container_width=True):
            st.session_state.page = "Advisories"
            st.rerun()


# =========================================================
# AI DIAGNOSIS PAGE
# =========================================================

elif st.session_state.page == "AI Diagnosis":

    st.markdown('<div class="krk-small-label">AI POWERED</div>', unsafe_allow_html=True)
    st.markdown("# Crop Disease Diagnosis")
    st.write("Upload a crop image and answer a few field questions before analysis.")

    col_upload, col_result = st.columns([1, 1.25])

    with col_upload:
        st.markdown('<div class="krk-card">', unsafe_allow_html=True)
        st.markdown("### 📷 Upload Crop Photo")
        st.write("Take a clear photograph of the affected plant or leaf.")

        uploaded_file = st.file_uploader("Choose Photo", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.session_state.uploaded_image = image
            st.image(image, width=220)

        analyze_disabled = uploaded_file is None

        if st.button("🔬 Analyze Crop", disabled=analyze_disabled, use_container_width=True):
            st.session_state.diag_step = 0
            st.session_state.diag_answers = {}
            st.session_state.show_questionnaire = True
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    # Re-open the dialog on every rerun for as long as it's flagged open —
    # this is what keeps it alive across "Next →" / "Back" clicks instead
    # of closing after the first question.
    if st.session_state.show_questionnaire:
        questionnaire_popup()

    with col_result:
        result = st.session_state.diag_result

        if result:
            disease_display = result["disease_display"]
            confidence = result["confidence"]
            answers = result["answers"]

            st.markdown('<div class="krk-card">', unsafe_allow_html=True)

            r1, r2 = st.columns([3, 1])
            with r1:
                st.markdown('<div class="krk-small-label">AI DIAGNOSIS</div>', unsafe_allow_html=True)
                st.markdown(f"## {disease_display}")
            with r2:
                if disease_display != "Healthy":
                    st.markdown('<span class="krk-severity-severe">DETECTED</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span style="color:#64d78a;font-weight:bold;">✅ HEALTHY</span>', unsafe_allow_html=True)

            st.markdown(f"**AI Confidence:** {confidence:.1f}%")
            st.progress(min(confidence / 100, 1.0))

            if disease_display != "Healthy":
                st.warning(
                    "⚠️ Symptoms indicate a possible crop disease. Follow the management advice "
                    "below and consider expert validation for severe or rapidly spreading cases."
                )

            st.markdown("""
            <div class="krk-treatment-box">
                <div class="krk-small-label">KNOWLEDGE LAYER</div>
                <h3>💊 Personalized Treatment Advice</h3>
                <p style="color:#8da79a;font-size:11px;">Advice generated using the AI diagnosis + your field answers.</p>
            </div>
            """, unsafe_allow_html=True)

            advice = generate_treatment_advice(disease_display, answers)

            for item in advice:
                st.markdown(f"""
                <div class="krk-advice-item">
                    <div class="krk-advice-icon">{item['icon']}</div>
                    <div>
                        <strong>{item['title']}</strong>
                        <p>{item['text']}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="krk-card" style="text-align:center;padding:60px 20px;color:#8ea99a;">
                Upload a photo and run analysis to see the diagnosis and treatment advice here.
            </div>
            """, unsafe_allow_html=True)


# =========================================================
# RISK FORECAST PAGE
# =========================================================

elif st.session_state.page == "Risk Forecast":

    st.markdown('<div class="krk-small-label">PREDICTIVE INTELLIGENCE</div>', unsafe_allow_html=True)
    st.markdown("# Disease Risk Forecast")
    st.write("Monitor environmental conditions that may increase crop disease risk.")

    f1, f2, f3, f4 = st.columns(4)
    for col, day, pct, label in [
        (f1, "Today", 72, "Late Blight"),
        (f2, "Tomorrow", 81, "Late Blight"),
        (f3, "Day 3", 65, "Fungal Risk"),
        (f4, "Day 4", 42, "Disease Risk"),
    ]:
        with col:
            st.markdown(f"""
            <div class="krk-card">
                <span style="color:#81998c;font-size:11px;">{day}</span>
                <h1 style="margin:8px 0;">{pct}%</h1>
                <p style="color:#99ad9f;font-size:12px;">{label}</p>
            </div>
            """, unsafe_allow_html=True)
            st.progress(pct / 100)

    st.markdown("""
    <div class="krk-card">
        <h3>🌧️ Why is the risk increasing?</h3>
        <p style="color:#9bb0a2;margin-top:10px;">
        High humidity, rainfall and prolonged leaf wetness can create favorable conditions
        for several crop diseases. Regular field scouting can help detect symptoms early.
        </p>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# HOTSPOT MAP PAGE
# =========================================================

elif st.session_state.page == "Hotspot Map":

    st.markdown('<div class="krk-small-label">GEOSPATIAL SURVEILLANCE</div>', unsafe_allow_html=True)
    st.markdown("# Disease Hotspot Map")
    st.write("Illustrative field-level disease surveillance.")

    hotspots = [
        {"name": "Belagavi Field", "disease": "Late Blight", "risk": "High"},
        {"name": "North Field", "disease": "Early Blight", "risk": "Medium"},
        {"name": "Village Farm", "disease": "Bacterial Spot", "risk": "High"},
        {"name": "South Field", "disease": "Healthy", "risk": "Low"},
    ]

    for h in hotspots:
        color = {"High": "#ff7770", "Medium": "#e4c55e", "Low": "#65d58a"}[h["risk"]]
        st.markdown(f"""
        <div class="krk-card" style="display:flex;justify-content:space-between;align-items:center;">
            <div>📍 <b>{h['name']}</b><br><span style="color:#99ad9f;font-size:12px;">Detected: {h['disease']}</span></div>
            <div style="color:{color};font-weight:bold;">{h['risk']}</div>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# FIELD MONITORING PAGE
# =========================================================

elif st.session_state.page == "Field Monitoring":

    st.markdown('<div class="krk-small-label">FIELD DATA</div>', unsafe_allow_html=True)
    st.markdown("# Field Monitoring")
    st.write("Track observations and disease progression.")

    if "monitoring_rows" not in st.session_state:
        st.session_state.monitoring_rows = [
            {"Field": "Field A", "Crop": "🍅 Tomato", "Observation": "Leaf spots", "Risk": "High", "Last Checked": "Today"},
            {"Field": "Field B", "Crop": "🥔 Potato", "Observation": "Brown lesions", "Risk": "Medium", "Last Checked": "Yesterday"},
            {"Field": "Field C", "Crop": "🌶️ Bell Pepper", "Observation": "No symptoms", "Risk": "Low", "Last Checked": "2 days ago"},
        ]

    if st.button("+ Add Observation"):
        st.session_state.monitoring_rows.insert(0, {
            "Field": "New Field", "Crop": "🍅 Tomato", "Observation": "New observation",
            "Risk": "Medium", "Last Checked": "Just now",
        })
        st.rerun()

    st.table(st.session_state.monitoring_rows)


# =========================================================
# ADVISORIES PAGE
# =========================================================

elif st.session_state.page == "Advisories":

    st.markdown('<div class="krk-small-label">INTEGRATED PEST MANAGEMENT</div>', unsafe_allow_html=True)
    st.markdown("# Crop Management Advisories")
    st.write("Practical actions for healthier crops.")

    advisories = [
        ("🔎", "Scout Regularly", "Inspect both healthy and affected plants regularly to detect changes early."),
        ("🌬️", "Reduce Leaf Wetness", "Improve field aeration and avoid unnecessary prolonged moisture on plant leaves."),
        ("🧹", "Field Sanitation", "Remove severely affected plant material where appropriate and maintain good field hygiene."),
        ("🧑‍🌾", "Expert Validation", "Seek agricultural extension or laboratory confirmation when symptoms are severe or uncertain."),
    ]

    a1, a2 = st.columns(2)
    for i, (icon, title, text) in enumerate(advisories):
        col = a1 if i % 2 == 0 else a2
        with col:
            st.markdown(f"""
            <div class="krk-card">
                <div style="font-size:28px;">{icon}</div>
                <h3 style="margin:10px 0 5px;">{title}</h3>
                <p style="color:#8fa89a;font-size:12px;line-height:1.7;">{text}</p>
            </div>
            """, unsafe_allow_html=True)
