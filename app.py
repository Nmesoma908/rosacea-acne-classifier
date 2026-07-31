import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

st.set_page_config(page_title="Rosacea vs Acne Classifier", page_icon="🩺")

@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model("skin_model.keras")
    with open("class_names.json") as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_model_and_classes()
IMAGE_HEIGHT, IMAGE_WIDTH = 128, 128

st.title("🩺 Rosacea vs Acne Classifier")
st.write("Upload a photo of the affected skin area to check whether it shows signs of Rosacea or Acne.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    resized = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    img_array = np.expand_dims(np.array(resized), axis=0)

    with st.spinner("Classifying..."):
        predictions = model.predict(img_array, verbose=0)[0]

    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = predictions[predicted_index] * 100

    st.subheader("Result")
    st.success(f"Prediction: **{predicted_class.capitalize()}** ({confidence:.2f}% confidence)")
    if confidence < 60:
        st.warning("Low confidence — result may be unreliable.")

    st.write("Class probabilities:")
    for name, prob in zip(class_names, predictions):
        st.write(f"- {name.capitalize()}: {prob*100:.2f}%")
