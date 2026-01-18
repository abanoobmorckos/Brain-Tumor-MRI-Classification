import tensorflow as tf
import numpy as np
from PIL import Image

# =========================
# Model Path
# =========================
MODEL_PATH = "models\MobileNetV2_STREAMLIT.keras"

# =========================
# Load Model (SAFE WAY)
# =========================
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

# =========================
# Class Names (عدّلهم حسب داتاستك)
# =========================
CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

# =========================
# Image Preprocessing
# =========================
IMG_SIZE = 224

def preprocess_image(image: Image.Image):
    """
    Preprocess image for MobileNetV2
    """
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(image)
    img_array = img_array / 255.0  # normalization

    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# =========================
# Prediction Function
# =========================
def predict_image(image: Image.Image):
    """
    Returns:
        predicted_class (str)
        confidence (float)
    """
    processed_image = preprocess_image(image)

    preds = model.predict(processed_image)
    class_index = np.argmax(preds)
    confidence = float(preds[0][class_index])

    predicted_class = CLASS_NAMES[class_index]

    return predicted_class, confidence

