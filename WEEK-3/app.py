import streamlit as st
import numpy as np
import joblib

# === Load model and scaler ===
model = joblib.load(r"C:\Users\venka\Desktop\AICTE_INERNSHIP_1\best_fire_detection_model.pkl")
scaler = joblib.load(r"C:\Users\venka\Desktop\AICTE_INERNSHIP_1\scaler.pkl")

# === Page configuration ===
st.set_page_config(page_title="🔥 Fire Type Classifier", layout="wide")

# === Custom CSS ===
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button {
        border-radius: 8px;
        background-color: #FF4B4B;
        color: white;
        font-size: 16px;
        height: 3em;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #D43F3F;
        transition: 0.3s;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #262730;
        color: white;
        font-size: 20px;
        text-align: center;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# === App title ===
st.markdown("<h1 style='text-align: center;'>🔥 Fire Type Classification App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict fire types using MODIS satellite features and a machine learning model</p>", unsafe_allow_html=True)
st.markdown("---")

# === Input Form ===
with st.form("prediction_form"):
    st.subheader("Enter Satellite Data:")

    col1, col2, col3 = st.columns(3)
    with col1:
        brightness = st.number_input("🔆 Brightness", value=300.0, step=1.0)
        frp = st.number_input("🔥 Fire Radiative Power (FRP)", value=15.0, step=1.0)

    with col2:
        bright_t31 = st.number_input("🌡️ Brightness T31", value=290.0, step=1.0)
        scan = st.number_input("📏 Scan", value=1.0, step=0.1)

    with col3:
        track = st.number_input("📍 Track", value=1.0, step=0.1)
        confidence = st.selectbox("📶 Confidence Level", ["low", "nominal", "high"])

    st.markdown("")
    submit_button = st.form_submit_button("🚀 Predict Fire Type")

# === Prediction Logic ===
if submit_button:
    confidence_map = {"low": 0, "nominal": 1, "high": 2}
    confidence_val = confidence_map[confidence]

    # Scale and predict
    input_data = np.array([[brightness, bright_t31, frp, scan, track, confidence_val]])
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]

    fire_types = {
        0: "🌿 Vegetation Fire",
        2: "🏜️ Other Static Land Source",
        3: "🌊 Offshore Fire"
    }

    result = fire_types.get(prediction, "❓ Unknown Type")

    # === Result Display ===
    st.markdown("<div class='result-box'>"
                f"<strong>🔥 Predicted Fire Type:</strong><br>{result}"
                "</div>", unsafe_allow_html=True)
