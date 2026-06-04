import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Gait Classification", layout="wide")

st.title("🚶 Gait Silhouette Classification")

@st.cache_resource
def load_models():
    cnn = tf.keras.models.load_model("cnn.keras")
    mob = tf.keras.models.load_model("mobilenet.keras")
    eff = tf.keras.models.load_model("effnet.keras")
    return cnn, mob, eff

cnn_model, mob_model, eff_model = load_models()

models = {
    "CNN": cnn_model,
    "MobileNetV2": mob_model,
    "EfficientNetB0": eff_model
}

model_name = st.selectbox("Choose Model", list(models.keys()))

file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

def preprocess(img):
    img = img.convert("RGB").resize((128,128))
    img = np.array(img)/255.0
    return np.expand_dims(img, axis=0)

if file:
    img = Image.open(file)
    st.image(img)

    model = models[model_name]

    pred = model.predict(preprocess(img))

    classes = ["Normal (nm)", "Bag (bg)", "Clothing (cl)"]

    st.success("Prediction: " + classes[np.argmax(pred)])

    st.bar_chart(pred[0])