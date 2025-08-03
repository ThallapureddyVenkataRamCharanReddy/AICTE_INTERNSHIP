
import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Fire Type Predictor", layout="centered", page_icon="🔥")

st.markdown("""
<style>
body {
    background-color: #0f1117;
    color: white;
}
.css-1aumxhk {
    padding-top: 1rem;
}
h1, h2, h3 {
    color: white;
}
.glass-card {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 2rem;
    margin-top: 2rem;
}
.stButton>button {
    background: linear-gradient(to right, #ff416c, #ff4b2b);
    color: white;
    font-size: 18px;
    border-radius: 12px;
    padding: 0.75rem 2rem;
    transition: 0.3s ease-in-out;
    margin-top: 1rem;
    width: 100%;
}
.stButton>button:hover {
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)


st.markdown("## 🔥 ML-Powered Fire Type Classifier")
st.markdown('<div class="glass-card">', unsafe_allow_html=True)

model = joblib.load("best_fire_detection_model.pkl")
scaler = joblib.load("scaler.pkl")

with st.form("prediction_form"):
    st.markdown("### 🛰️ Enter Satellite Data")

    col1, col2, col3 = st.columns(3)
    with col1:
        brightness = st.number_input("🌞 Brightness", min_value=200.0, max_value=500.0, value=300.0)
    with col2:
        bright_t31 = st.number_input("🌡️ Brightness T31", min_value=200.0, max_value=500.0, value=290.0)
    with col3:
        track = st.number_input("📍 Track", min_value=0.1, max_value=10.0, value=1.0)
    
    col4, col5, col6 = st.columns(3)
    with col4:
        frp = st.number_input("🔥 Fire Radiative Power (FRP)", min_value=0.0, max_value=500.0, value=15.0)
    with col5:
        scan = st.number_input("🧪 Scan", min_value=0.1, max_value=10.0, value=1.0)
    with col6:
        confidence = st.selectbox("📊 Confidence Level", ["low", "nominal", "high"])

    submit = st.form_submit_button("🚀 Predict Fire Type")

st.markdown('</div>', unsafe_allow_html=True)

# Prediction Logic
if submit:
    confidence_map = {"low": 0, "nominal": 1, "high": 2}
    confidence_val = confidence_map[confidence]

    input_data = np.array([[brightness, bright_t31, track, frp, scan, confidence_val]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    # Fire type interpretation
    fire_types = {
        0: "Vegetation Fire",
        2: "Other Static Land Source",
        3: "Offshore Fire"
    }

    result = fire_types.get(prediction, "Unknown")

    st.markdown("---")
    st.subheader("🎯 Prediction Result")

    if result == "Vegetation Fire":
        st.success("🌲🔥 **Vegetation Fire Detected!** Take immediate action.")
    elif result == "Other Static Land Source":
        st.warning("🌆🔥 **Other Static Land Source Detected.** May not need urgent response.")
    elif result == "Offshore Fire":
        st.info("🌊🔥 **Offshore Fire Detected.** Monitor coastal areas.")
    else:
        st.error("⚠️ **Unknown Fire Type Detected.** Please verify inputs.")
