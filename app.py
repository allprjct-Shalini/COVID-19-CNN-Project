import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Set page title and layout
st.set_page_config(page_title="COVID-19 Chest X-Ray Detector", page_icon="🦠", layout="centered")

# Cache the model loading to improve performance
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('best_covid_model.h5')
    return model

with st.spinner("Loading AI Model... Please wait."):
    model = load_model()

st.title("🦠 COVID-19 Detection from Chest X-Ray")
st.write("Upload a chest X-ray image to check for COVID-19, Viral Pneumonia, or Normal lungs using Deep Learning.")

# File uploader widget
uploaded_file = st.file_uploader("Choose a Chest X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded X-Ray Image', use_column_width=True)
    
    # Preprocess image for prediction
    image_resized = image.resize((150, 150))
    img_array = np.array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    if st.button('Predict Disease'):
        with st.spinner('Analyzing scan...'):
            prediction = model.predict(img_array)
            classes = ['Covid', 'Normal', 'Viral Pneumonia']
            predicted_class = classes[np.argmax(prediction)]
            confidence = np.max(prediction) * 100
            
        st.success(f"### Prediction Result: **{predicted_class}**")
        st.info(f"### Confidence Score: **{confidence:.2f}%**")
        
        # Display breakdown
        st.write("#### Detailed Probabilities:")
        for idx, cls in enumerate(classes):
            st.write(f"- **{cls}**: {prediction[0][idx]*100:.2f}%")