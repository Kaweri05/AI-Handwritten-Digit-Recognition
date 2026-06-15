import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ✅ FIX: safe model loading for Streamlit
model = tf.keras.models.load_model("digit_model.h5", compile=False)

st.title("Handwritten Digit Recognition")

uploaded_file = st.file_uploader(
    "Upload Digit Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("L")

    image = image.resize((28, 28))

    img = np.array(image)

    img = img / 255.0

    img = img.reshape(1, 28, 28, 1)

    prediction = model.predict(img)

    digit = np.argmax(prediction)

    st.image(image, width=200)

    st.success(f"Predicted Digit: {digit}")
