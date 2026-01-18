import tensorflow as tf
import numpy as np
from PIL import Image

# =====================
# Model Path
# =====================
MODEL_PATH = "models/MobileNetV2_CLEAN.keras"

# =====================
# Load Model (مرة واحدة)
# =====================
model = tf.keras.models.load_model(MODEL_PATH)

# =====================
# Class Names (لازم نفس ترتيب التدريب)
# =====================
class_names = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

# =====================
# Prediction Function
# =====================
def predict_image(image_file):
    # فتح الصورة
    image = Image.open(image_file).convert("RGB")

    # Resize (MobileNetV2 input)
    image = image.resize((224, 224))

    # تحويل لـ numpy
    image = np.array(image) / 255.0

    # إضافة batch dimension
    image = np.expand_dims(image, axis=0)

    # Predict
    predictions = model.predict(image)

    # استخراج النتيجة
    predicted_index = np.argmax(predictions)
    predicted_label = class_names[predicted_index]
    confidence = float(np.max(predictions) * 100)

    return predicted_label, confidence
