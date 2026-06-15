import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# rebuild model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# load weights only
model.load_weights("digit_weights.weights.h5")

st.title("Handwritten Digit Recognition")

uploaded_file = st.file_uploader("Upload Digit Image", type=["png","jpg","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    image = image.resize((28,28))

    img = np.array(image) / 255.0
    img = img.reshape(1,28,28,1)

    prediction = model.predict(img)
    digit = np.argmax(prediction)

    st.image(image, width=200)
    st.success(f"Predicted Digit: {digit}")
