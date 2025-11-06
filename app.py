import streamlit as st                                ### Streamlit → Web app UI
import tensorflow as tf                               ### Load trained ANN model
from tensorflow.keras.models import load_model        ### For loading saved .h5 model
import numpy as np                                    ### Convert image into numeric format
from PIL import Image                                 ### For image processing
from streamlit_drawable_canvas import st_canvas       ### Drawing canvas

# Load trained Fashion model
model = load_model("fashion_ann_model.h5")            ### Load your trained model file

# Label names for Fashion MNIST
fashion_labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                  "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]

# Title

st.title("👗👜 Fashion Item Recognizer (ANN) - By Satyam")

st.write("✍️ Draw a fashion item like **shoe, bag, shirt etc.**, then click Predict")

# Drawing Canvas
canvas_result = st_canvas(
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

# Predict button
if st.button("Predict"):
    if canvas_result.image_data is not None:
        img = canvas_result.image_data

        # Convert to grayscale & preprocess
        img = Image.fromarray((img).astype("uint8")).convert("L")   ### Grayscale
        img = img.resize((28, 28))                                  ### Resize to model size
        img_arr = np.array(img) / 255.0                             ### Normalize
        img_arr = img_arr.reshape(1, 28, 28)                        ### Reshape for ANN

        # Predict
        prediction = model.predict(img_arr)
        label = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        # Show result
        st.write(f"### ✅ Predicted Item: **{fashion_labels[label]}**")
        st.write(f"### 🎯 Confidence: **{confidence:.2f}%**")

    else:
        st.warning("⚠️ Please draw something before predicting.")
