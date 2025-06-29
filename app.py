import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Load model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("pattern_sense_model.keras")

model = load_model()

# Load index-to-label mapping
with open("index_to_label.json") as f:
    index_to_label = json.load(f)
index_to_label = {int(k): v for k, v in index_to_label.items()}

# Page styling
st.set_page_config(page_title="Pattern Sense", page_icon="🧵", layout="centered")
st.markdown(
    "<h1 style='text-align: center; color: #6A5ACD;'>🧵 Pattern Sense: Fabric Pattern Classifier</h1>",
    unsafe_allow_html=True
)
st.markdown("<p style='text-align: center;'>Upload a fabric image to classify its pattern using AI!</p>", unsafe_allow_html=True)
st.markdown("---")

# Upload box
uploaded_file = st.file_uploader("📂 Upload a fabric image", type=["jpg", "jpeg", "png"])

# Two-column layout
col1, col2 = st.columns([1, 2])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    col1.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        with st.spinner("🔍 Analyzing the fabric pattern..."):
            # Preprocess image
            img = image.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            prediction = model.predict(img_array)
            predicted_index = np.argmax(prediction) + 1
            predicted_label = index_to_label.get(predicted_index, "Unknown")

        st.markdown("### ✅ Prediction Result")
        st.success(f"🧠 Predicted Pattern: **{predicted_label}**")
        st.markdown("### 🔢 Raw Prediction Scores")
        st.code(prediction.tolist())

else:
    st.info("📸 Please upload a fabric image to get started.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; font-size: 14px;'>Made with 💜 using TensorFlow and Streamlit</div>",
    unsafe_allow_html=True
)
