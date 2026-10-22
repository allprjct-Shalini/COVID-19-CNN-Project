import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Page Configuration
st.set_page_config(page_title="COVID-19 Detection from X-rays", page_icon="🩺", layout="centered")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('covid_resnet_model.h5')
    return model

with st.spinner('Loading AI Model... Please wait!'):
    model = load_model()

st.title("🩺 COVID-19 Detection from Chest X-Rays")
st.write("Upload a chest X-ray image below to check for Normal, COVID-19, or Viral Pneumonia classifications.")

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded X-Ray Image', use_column_width=True)
    
    # Preprocess Image
    img = np.array(image)
    if len(img.shape) == 2: # Convert grayscale to RGB if needed
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4: # Remove alpha channel if PNG
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
    img_resized = cv2.resize(img, (128, 128))
    img_normalized = img_resized.astype('float32') / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=0)
    
    if st.button('Predict Disease'):
        prediction = model.predict(img_expanded)
        class_idx = np.argmax(prediction[0])
        confidence = float(np.max(prediction[0])) * 100
        
        categories = ['Normal', 'COVID-19', 'Viral Pneumonia']
        result = categories[class_idx]
        
        st.subheader("Prediction Results:")
        if result == "COVID-19":
            st.error(f"⚠️ Prediction: **{result}** (Confidence: {confidence:.2f}%)")
        elif result == "Viral Pneumonia":
            st.warning(f"⚠️ Prediction: **{result}** (Confidence: {confidence:.2f}%)")
        else:
            st.success(f"✅ Prediction: **{result}** (Confidence: {confidence:.2f}%)")