import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Page Configuration
st.set_page_config(page_title="COVID-19 Chest X-Ray Detector", page_icon="🩻", layout="centered")

st.title("🩻 COVID-19 Detection from Chest X-Rays")
st.write("Upload a chest X-ray image below to automatically analyze and detect Normal, Viral Pneumonia, or COVID-19 infection.")

# Load Model safely with caching
@st.cache_resource
def load_prediction_model():
    model = tf.keras.models.load_model('covid_model.h5')
    return model

try:
    model = load_prediction_model()
except Exception as e:
    st.error(f"Error loading model: {e}. Please ensure 'covid_model.h5' is in the same folder as app.py.")

# File uploader widget
uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded X-Ray Image', use_column_width=True)
    
    if st.button('Run Prediction'):
        with st.spinner('Analyzing scan... Please wait.'):
            # Preprocess the image
            image = image.resize((128, 128))
            img_array = np.array(image)
            
            # Handle grayscale or 4-channel RGBA images cleanly
            if len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.shape[2] == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            predictions = model.predict(img_array)
            class_idx = np.argmax(predictions)
            confidence = float(np.max(predictions)) * 100
            
            # Dataset classes mapping alphabetical order based on folder names (Covid, Normal, Viral Pneumonia)
            class_labels = ['Covid', 'Normal', 'Viral Pneumonia']
            result = class_labels[class_idx]
            
            # Display results
            st.markdown("---")
            st.subheader("Diagnostic Results:")
            if result == 'Covid':
                st.error(f"**Prediction:** COVID-19 Positive\n\n**Confidence:** {confidence:.2f}%")
            elif result == 'Viral Pneumonia':
                st.warning(f"**Prediction:** Viral Pneumonia\n\n**Confidence:** {confidence:.2f}%")
            else:
                st.success(f"**Prediction:** Normal (No Abnormality)\n\n**Confidence:** {confidence:.2f}%")
