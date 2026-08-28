import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import json
import requests
import os
import pandas as pd
from datetime import datetime
from sklearn.cluster import DBSCAN
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

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
    .stApp .stSelectbox label, .stApp .stFileUploader label, .stApp .stTextInput label { color: #eef2e6 !important; }

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
    .weather-risk-box {
        background: rgba(120,170,255,0.08);
        border-radius: 18px;
        padding: 1.3rem 1.6rem;
        margin-top: 1.2rem;
        border: 1px solid rgba(120,170,255,0.30);
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
    .weather-card {
        background: rgba(255,255,255,0.06);
        border-radius: 18px;
        padding: 1.2rem 1.4rem;
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow: 0 4px 16px rgba(0,0,0,0.25);
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
    .hotspot-card {
        background: rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 1rem 1.2rem;
        border-left: 5px solid #e0665a;
        margin-bottom: 10px;
    }

    div[data-testid="stSelectbox"] > div {
        background: rgba(255,255,255,0.08) !important;
        border-radius: 14px !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
    }
    div[data-testid="stSelectbox"] div, div[data-testid="stSelectbox"] span {
        color: #eef2e6 !important;
    }
    div[data-testid="stTextInput"] input {
        background: rgba(255,255,255,0.08) !important;
        border-radius: 14px !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
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

# =====================================================================================
#  PHASE 2 — SMART PREDICTION: Image + Weather -> Disease Risk Forecast
#  LOCATION + WEATHER (automatic — no manual entry required)
#  Location: IP-based geolocation (ipapi.co)
#  Weather:  Open-Meteo (satellite/reanalysis-backed forecast model, free, no API key)
# =====================================================================================

@st.cache_data(ttl=1800, show_spinner=False)
def get_location_from_ip():
    """Auto-detect the user's approximate location from their IP address."""
    try:
        r = requests.get("https://ipapi.co/json/", timeout=5)
        r.raise_for_status()
        d = r.json()
        if d.get("latitude") is None:
            return None
        return {
            "lat": d.get("latitude"),
            "lon": d.get("longitude"),
            "city": d.get("city", "Unknown"),
            "region": d.get("region", ""),
            "source": "auto (IP-based)"
        }
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def geocode_city(city_name):
    """Fallback: resolve a manually typed city name to coordinates."""
    try:
        r = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city_name, "count": 1},
            timeout=5
        )
        r.raise_for_status()
        results = r.json().get("results")
        if not results:
            return None
        top = results[0]
        return {
            "lat": top["latitude"],
            "lon": top["longitude"],
            "city": top.get("name", city_name),
            "region": top.get("admin1", ""),
            "source": "manual (typed location)"
        }
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def get_weather(lat, lon):
    """Pull current temperature, humidity and precipitation for the field's location."""
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
                "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min",
                "timezone": "auto",
                "forecast_days": 1,
            },
            timeout=6
        )
        r.raise_for_status()
        d = r.json()
        cur = d.get("current", {})
        daily = d.get("daily", {})
        return {
            "temp": cur.get("temperature_2m"),
            "humidity": cur.get("relative_humidity_2m"),
            "precip": cur.get("precipitation"),
            "wind": cur.get("wind_speed_10m"),
            "rain_today": (daily.get("precipitation_sum") or [None])[0],
            "temp_max": (daily.get("temperature_2m_max") or [None])[0],
            "temp_min": (daily.get("temperature_2m_min") or [None])[0],
        }
    except Exception:
        return None


def risk_level(score):
    """Map a 0-1 environmental favorability score to a level, color, label."""
    if score >= 0.75:
        return "#e0665a", "Critical — conditions strongly favor spread"
    elif score >= 0.5:
        return "#f2914a", "High — conditions favor spread"
    elif score >= 0.25:
        return "#f2c744", "Moderate — some risk of spread"
    else:
        return "#7bd389", "Low — conditions currently unfavorable for spread"


def compute_environmental_risk(info, weather):
    """
    Score (0-1) how favorable current weather is for THIS disease to spread,
    based on its known ideal temperature range and minimum humidity.
    """
    if not weather or weather.get("temp") is None or weather.get("humidity") is None:
        return None
    tmin, tmax = info.get("ideal_temp_range", (15, 30))
    hmin = info.get("ideal_humidity_min", 70)
    temp = weather["temp"]
    humidity = weather["humidity"]

    if tmin <= temp <= tmax:
        temp_score = 1.0
    else:
        dist = min(abs(temp - tmin), abs(temp - tmax))
        temp_score = max(0.0, 1 - dist / 8.0)

    humidity_score = min(1.0, humidity / hmin) if hmin else 1.0

    # extra push if it has rained recently — most fungal/bacterial spread needs moisture on leaves
    rain_bonus = 0.15 if (weather.get("rain_today") or 0) > 1 else 0.0

    score = min(1.0, temp_score * 0.5 + humidity_score * 0.4 + rain_bonus)
    return score


# ---------- Resolve location (automatic first, manual override optional) ----------
if "location" not in st.session_state:
    auto_loc = get_location_from_ip()
    st.session_state["location"] = auto_loc  # may be None if lookup failed

with st.expander("📍 Detected location (auto) — click to correct if wrong", expanded=False):
    loc = st.session_state.get("location")
    if loc:
        st.caption(f"Using **{loc['city']}, {loc['region']}** ({loc['source']}) for weather-based risk.")
    else:
        st.caption("Could not auto-detect location — please type your city/district below.")
    manual_city = st.text_input("Correct city / district (optional)", value="", placeholder="e.g. Nagpur, Maharashtra")
    if manual_city:
        corrected = geocode_city(manual_city)
        if corrected:
            st.session_state["location"] = corrected
            st.success(f"Location updated to {corrected['city']}, {corrected['region']}")
        else:
            st.warning("Couldn't find that location — keeping previous location.")

location = st.session_state.get("location")
weather = get_weather(location["lat"], location["lon"]) if location else None

# =====================================================================
# PHASE 3 — DISEASE SURVEILLANCE (Field Reports + GIS -> Hotspot Detection)
# Reuses the SAME auto-detected location above, so every submitted
# report carries real coordinates instead of a manually picked district.
# =====================================================================
REPORTS_FILE = "field_reports.csv"
HOTSPOT_EPS_KM = 25.0          # ~district-scale radius for clustering
HOTSPOT_MIN_REPORTS = 3        # min reports before it counts as a real hotspot
EARTH_RADIUS_KM = 6371.0


def load_reports():
    if os.path.exists(REPORTS_FILE):
        return pd.read_csv(REPORTS_FILE, parse_dates=["timestamp"])
    return pd.DataFrame(columns=[
        "timestamp", "place", "latitude", "longitude",
        "disease", "severity", "confidence"
    ])


def log_field_report(place, lat, lon, disease, severity, confidence):
    df = load_reports()
    new_row = {
        "timestamp": datetime.now(),
        "place": place,
        "latitude": lat,
        "longitude": lon,
        "disease": disease,
        "severity": severity,
        "confidence": confidence,
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(REPORTS_FILE, index=False)


def detect_hotspots(df):
    """DBSCAN clustering on report coordinates -> district-level hotspots."""
    if len(df) < HOTSPOT_MIN_REPORTS:
        return pd.DataFrame()

    coords_rad = np.radians(df[["latitude", "longitude"]].values)
    eps_rad = HOTSPOT_EPS_KM / EARTH_RADIUS_KM
    db = DBSCAN(eps=eps_rad, min_samples=HOTSPOT_MIN_REPORTS, metric="haversine").fit(coords_rad)
    df = df.copy()
    df["cluster_id"] = db.labels_

    hotspots = []
    severity_score = {"healthy": 1, "moderate": 5, "severe": 9}
    for cid in sorted(set(db.labels_)):
        if cid == -1:
            continue
        cluster_df = df[df["cluster_id"] == cid]
        avg_sev = cluster_df["severity"].map(severity_score).mean()
        hotspots.append({
            "hotspot_id": int(cid),
            "area(s)": ", ".join(sorted(cluster_df["place"].unique())),
            "report_count": len(cluster_df),
            "center_lat": cluster_df["latitude"].mean(),
            "center_lon": cluster_df["longitude"].mean(),
            "dominant_disease": cluster_df["disease"].mode()[0],
            "risk_level": "HIGH" if avg_sev >= 7 else "MEDIUM" if avg_sev >= 4 else "LOW",
        })
    return pd.DataFrame(hotspots).sort_values("report_count", ascending=False)


def build_surveillance_map(df, hotspots):
    center = [df["latitude"].mean(), df["longitude"].mean()]
    m = folium.Map(location=center, zoom_start=6, tiles="CartoDB positron")

    HeatMap(df[["latitude", "longitude"]].values.tolist(), radius=20).add_to(m)

    risk_colors = {"HIGH": "red", "MEDIUM": "orange", "LOW": "green"}
    for _, h in hotspots.iterrows():
        folium.CircleMarker(
            location=[h["center_lat"], h["center_lon"]],
            radius=8 + h["report_count"] * 0.6,
            color=risk_colors.get(h["risk_level"], "orange"),
            fill=True,
            fill_opacity=0.75,
            popup=(f"Area: {h['area(s)']}<br>"
                   f"Reports: {h['report_count']}<br>"
                   f"Disease: {h['dominant_disease']}<br>"
                   f"Risk: {h['risk_level']}"),
        ).add_to(m)
    return m


# ---------- Hero ----------
st.markdown("""
    <div class="hero">
        <div class="hero-title">🌾 KrishiRakshak AI</div>
        <div class="hero-sub">Early Detection & Weather-Aware Risk Assessment for Crop Diseases and Pest Infestations</div>
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
    st.markdown('<div class="stat-card"><div class="stat-num">Live</div><div class="stat-label">Weather-linked risk</div></div>', unsafe_allow_html=True)

st.write("")

# ---------- Live weather strip ----------
st.markdown('<div class="section-header">🌦️ Live field conditions</div>', unsafe_allow_html=True)
if weather and location:
    w1, w2, w3, w4, w5 = st.columns(5)
    with w1:
        st.markdown(f'<div class="weather-card"><div class="stat-num">📍</div><div class="stat-label">{location["city"]}, {location["region"]}</div></div>', unsafe_allow_html=True)
    with w2:
        st.markdown(f'<div class="weather-card"><div class="stat-num">{weather["temp"]:.1f}°C</div><div class="stat-label">Temperature</div></div>', unsafe_allow_html=True)
    with w3:
        st.markdown(f'<div class="weather-card"><div class="stat-num">{weather["humidity"]:.0f}%</div><div class="stat-label">Humidity</div></div>', unsafe_allow_html=True)
    with w4:
        rain_val = weather.get("rain_today")
        st.markdown(f'<div class="weather-card"><div class="stat-num">{rain_val if rain_val is not None else "–"} mm</div><div class="stat-label">Rain today</div></div>', unsafe_allow_html=True)
    with w5:
