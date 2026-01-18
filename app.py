import streamlit as st
from predict import predict_image
from PIL import Image

# =====================
# Page Config
# =====================
st.set_page_config(
    page_title="Brain Tumor MRI Classification",
    page_icon="🧠",
    layout="wide"
)

# =====================
# Custom CSS
# =====================
st.markdown("""
<style>
body {
    background-color: #f5f7fb;
}

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    color: #1f3c88;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #6c757d;
    margin-bottom: 30px;
}

.card {
    background-color: white;
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.08);
}

.result-good {
    border-left: 6px solid #2ecc71;
}

.predict-btn {
    background-color: #1f3c88;
    color: white;
    border-radius: 10px;
    padding: 0.6em 1.2em;
    font-size: 18px;
}

footer {
    text-align: center;
    font-size: 12px;
    color: gray;
}
</style>
""", unsafe_allow_html=True)

# =====================
# Header
# =====================
st.markdown("<div class='main-title'>🧠 Brain Tumor MRI Classification</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Upload an MRI image and get tumor type prediction using AI</div>",
    unsafe_allow_html=True
)

# =====================
# Layout
# =====================
left, right = st.columns([1.2, 1])

# =====================
# Upload Section
# =====================
with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "📤 Upload MRI Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI", use_column_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# =====================
# Prediction Section
# =====================
with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.markdown("### 🧪 Prediction Result")

    if uploaded_file:
        if st.button("🔍 Analyze MRI"):
            with st.spinner("Analyzing MRI image..."):
                predicted_class, confidence = predict_image(uploaded_file)

            st.markdown(
                f"""
                <div class="card result-good">
                    <h2>🧠 {predicted_class}</h2>
                    <h4>📊 Confidence: {confidence:.2f}%</h4>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("⬅️ Please upload an MRI image to start analysis.")

    st.markdown("</div>", unsafe_allow_html=True)

# =====================
# Footer
# =====================
st.markdown("---")
st.markdown(
    "<footer>⚠️ This application is for research and educational purposes only. Not for clinical diagnosis.</footer>",
    unsafe_allow_html=True
)
