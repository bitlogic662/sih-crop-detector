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
        background: linear-gradient(135deg, #14532d 0%, #2d6a4f 35%, #40916c 65%, #74c69d 100%);
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
        opacity: 0.14;
        transform: rotate(15deg);
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ffffff, #ffe066, #d8f3dc);
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
        border-radius: 14px;
        padding: 1rem 1.2rem;
        text-align: center;
        box-shadow: 0 3px 12px rgba(0,0,0,0.08);
        border: 1px solid rgba(255,255,255,0.4);
        transition: transform 0.2s ease;
        color: white;
    }
    .stat-card:hover { transform: translateY(-3px); }
    .stat-num {
        font-size: 1.7rem;
        font-weight: 800;
        color: white;
    }
    .stat-label { font-size: 0.8rem; color: rgba(255,255,255,0.9); margin-top: 2px; }
    .stat-1 { background: linear-gradient(135deg, #1b4332, #40916c); }
    .stat-2 { background: linear-gradient(135deg, #2d6a4f, #74c69d); }
    .stat-3 { background: linear-gradient(135deg, #8a5a2e, #c98a3f); }
    .stat-4 { background: linear-gradient(135deg, #52734d, #95b46a); }

    .upload-panel-label {
        font-weight: 700;
        color: #14532d;
        margin-bottom: 0.6rem;
    }

    /* Style Streamlit's ACTUAL dropzone so the whole visible box is the real drop target */
    div[data-testid="stFileUploaderDropzone"] {
        border: 2.5px dashed #40916c !important;
        border-radius: 18px !important;
        background: linear-gradient(135deg, #f0fdf4 0%, #e6f7ee 100%) !important;
        padding: 1.2rem !important;
        box-shadow: 0 3px 14px rgba(45,106,79,0.08);
        transition: border-color 0.2s ease, background 0.2s ease;
    }
    div[data-testid="stFileUploaderDropzone"]:hover {
        border-color: #c98a3f !important;
        background: linear-gradient(135deg, #fdf6ea 0%, #faecd2 100%) !important;
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
