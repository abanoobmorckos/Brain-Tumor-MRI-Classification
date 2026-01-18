import streamlit as st
from predict import predict_image

# =====================
# Page Config
# =====================
st.set_page_config(
    page_title="Brain Tumor MRI Classification",
    layout="centered"
)

# =====================
# Title
# =====================
st.title("🧠 Brain Tumor MRI Classification")
st.write("Upload an MRI image and get tumor type prediction")

# =====================
# File Upload
# =====================
uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

# =====================
# Prediction
# =====================
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded MRI", use_column_width=True)

    with st.spinner("Analyzing MRI Image..."):
        prediction, confidence = predict_image(uploaded_file)

    st.success(f"🧪 **Prediction:** {prediction}")
    st.info(f"📊 **Confidence:** {confidence:.2f}%")

# =====================
# Disclaimer
# =====================
st.warning(
    "⚠️ Disclaimer: This application is for research and educational purposes only. "
    "It is not intended for clinical diagnosis."
)
