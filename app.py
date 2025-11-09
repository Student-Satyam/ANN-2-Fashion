import streamlit as st
import tensorflow as tf
import numpy as np
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# ✅ Load trained model (.keras)
model = tf.keras.models.load_model("fashion_ann_model.keras")

# ✅ Class names for Fashion MNIST
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# ✅ Streamlit page setup
st.set_page_config(page_title="🖌️ Fashion Classifier", layout="centered")

st.title("👕 Draw a Fashion Item")
st.write("Draw a clothing item below (like a shoe, shirt, or bag) and the model will predict what it is!")

# ✅ Create drawing canvas
canvas_result = st_canvas(
    fill_color="white",  # White background
    stroke_width=8,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# ✅ Predict button
if st.button("🧠 Predict"):
    if canvas_result.image_data is not None:
        # Convert canvas (RGBA) to grayscale PIL image
        img = Image.fromarray((255 - canvas_result.image_data[:, :, 0]).astype('uint8'))  # invert so black ink = high intensity
        img = img.convert("L").resize((28, 28))

        # Preprocess image for model
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28)

        # Predict
        predictions = model.predict(img_array)
        pred_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100

        st.subheader("🎯 Prediction Result:")
        st.success(f"**Predicted Class:** {class_names[pred_class]}")
        st.info(f"**Confidence:** {confidence:.2f}%")
    else:
        st.warning("Please draw something before predicting!")
