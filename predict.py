import tensorflow as tf
import numpy as np
from PIL import Image

# اسم الموديل و مكانه
MODEL_PATH = "models\MobileNetV2_STREAMLIT.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# أسماء الكلاسات
CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

# دالة لتحويل الصورة وتحضيرها للموديل
def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224, 224))  # حسب input_shape للموديل
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# دالة التنبؤ
def predict_image(uploaded_file):
    image = preprocess_image(uploaded_file)
    prediction = model.predict(image)
    predicted_class_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    return predicted_class_name, confidence



