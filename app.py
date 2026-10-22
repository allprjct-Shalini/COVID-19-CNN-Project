import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="COVID-19 Image Classification",
    page_icon="🦠",
    layout="centered"
)

# App Header
st.title("🦠 COVID-19 Chest X-Ray Classification")
st.write("Upload a chest X-ray or scan image to analyze it using the trained deep learning model.")

# Load Model with Caching for Performance
@st.cache_resource
def load_model():
    # Loads the model (Ensure 'include_optimizer=False' was used when saving to keep file size small)
    model = tf.keras.models.load_model("covid_model.h5")
    return model

with st.spinner("Loading AI model... Please wait."):
    model = load_model()

# File Uploader Widget
file = st.file_uploader("Upload a chest scan image (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

if file is not None:
    # Display the uploaded image
    image = Image.open(file)
    st.image(image, caption="Uploaded Scan", use_container_width=True)
    
    # Preprocess the image to match model input shape (128, 128, 3)
    image = ImageOps.fit(image, (128, 128), Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    
    # Normalize pixel values (matches typical training scaling / 255.0)
    img_array = img_array / 255.0
    
    # Handle RGBA to RGB conversion if the image has an alpha channel
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
        
    # Expand dimensions to create a batch size of 1 -> shape becomes (1, 128, 128, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction Button
    if st.button("Predict"):
        with st.spinner("Analyzing scan..."):
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            confidence = 100 * np.max(predictions[0])
            
            # TODO: Update these labels to match the exact alphabetical/numerical order 
            # of your training generator classes (e.g., ['COVID', 'Normal', 'Viral Pneumonia'])
            class_names = ["Class 0", "Class 1", "Class 2"]
            
            # Display Results
            st.markdown("---")
            st.subheader("Analysis Results")
            st.success(f"**Prediction:** {class_names[predicted_class]}")
            st.info(f"**Confidence Score:** {confidence:.2f}%")
            
            # Optional: Show breakdown of all classes
            with st.expander("See full probability breakdown"):
                for i, class_name in enumerate(class_names):
                    st.write(f"{class_name}: {predictions[0][i] * 100:.2f}%")