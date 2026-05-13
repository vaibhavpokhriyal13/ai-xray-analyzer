import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# Page settings
st.set_page_config(
    page_title="AI X-Ray Analyzer",
    page_icon="🩻",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    background-color: #FF4B4B;
    color: white;
    font-size: 18px;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    background-color: #1E1E1E;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# Load and cache model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

IMG_SIZE = 224

# Sidebar
st.sidebar.title("🩻 AI X-Ray Analyzer")
st.sidebar.markdown("""
### About
This AI model analyzes chest X-rays and predicts possible pneumonia infection.

### Features
- Deep Learning Detection
- Confidence Score
- AI Report
- Fast Prediction

### Disclaimer
For educational purposes only.
Not medical advice.
""")

# Main title
st.title("🩻 AI-Powered Chest X-Ray Analyzer")
st.markdown("Upload a chest X-ray image for AI-based pneumonia detection.")

# Upload section
uploaded_file = st.file_uploader(
    "Upload X-Ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    col1, col2 = st.columns(2)

    # Display image
    with col1:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded X-Ray", width=400)

    # Prediction section
    with col2:

        if st.button("Analyze X-Ray"):

            with st.spinner("Analyzing X-Ray..."):

                time.sleep(2)

                # Preprocess image
                img = image.resize((IMG_SIZE, IMG_SIZE))
                img = np.array(img) / 255.0
                img = np.expand_dims(img, axis=0)

                # Prediction
                prediction = model.predict(img)[0][0]

                st.markdown("<div class='result-box'>", unsafe_allow_html=True)

                # Threshold
                if prediction > 0.8:

                    confidence = prediction * 100

                    st.error("⚠ Pneumonia Detected")

                    st.progress(int(confidence))

                    st.write(f"### Confidence: {confidence:.2f}%")

                    st.markdown("""
                    ## AI Medical Report
                    - Possible lung infection detected
                    - Abnormal opacity patterns observed
                    - Clinical evaluation recommended
                    """)

                else:

                    confidence = (1 - prediction) * 100

                    st.success("✅ Normal Chest X-Ray")

                    st.progress(int(confidence))

                    st.write(f"### Confidence: {confidence:.2f}%")

                    st.markdown("""
                    ## AI Medical Report
                    - No major pneumonia indicators detected
                    - Lung structure appears normal
                    - No obvious infection patterns observed
                    """)

                st.markdown("</div>", unsafe_allow_html=True)