import streamlit as st
from predict import predict_image
from PIL import Image

# =====================
# Page Config
# =====================
st.set_page_config(
    page_title="🧠 Brain Tumor MRI Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================
# Header
# =====================
st.markdown(
    """
    <h1 style='text-align: center; color: darkblue; font-family: Arial;'>🧠 Brain Tumor MRI Classification</h1>
    <p style='text-align: center; font-size:16px; color:gray; font-family: Arial;'>
    Upload an MRI image and get tumor type prediction using our fine-tuned MobileNetV2 model.
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# =====================
# File Upload
# =====================
uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible"
)

if uploaded_file:
    # عرض الصورة المرفوعة
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI", use_column_width=True)

    # Predict button
    if st.button("🧪 Predict"):
        with st.spinner("Analyzing MRI..."):
            predicted_class_name, confidence = predict_image(uploaded_file)

        # =====================
        # Result Display
        # =====================
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(image, caption="MRI Image", use_column_width=True)
        with col2:
            st.markdown(
                f"""
                <div style='border:2px solid #4CAF50; padding:20px; border-radius:15px; background-color:#f0f8ff'>
                    <h2 style='color:#2E8B57;'>Prediction: {predicted_class_name}</h2>
                    <h4 style='color:#555555;'>Confidence: {confidence:.2f}%</h4>
                </div>
                """,
                unsafe_allow_html=True
            )

st.markdown("---")

# =====================
# Disclaimer
# =====================
st.markdown(
    "<sub>⚠️ This application is for research and educational purposes only. Not intended for clinical diagnosis.</sub>",
    unsafe_allow_html=True
)
