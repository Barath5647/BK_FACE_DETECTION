import io
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from typing import Tuple

# -----------------------
# CONFIG / CONSTANTS
# -----------------------
IMG_SIZE = (160, 160)                       # same as training
MODEL_PATH = "best_model.h5"                # your trained model
EMBEDDINGS_PATH = "embeddings.npy"          # saved embeddings
LABELS_PATH = "labels.npy"                  # saved labels

# -----------------------
# LOAD MODEL + DATA
# -----------------------
@st.cache_resource
def load_face_model():
    return load_model(MODEL_PATH, compile=False)

@st.cache_resource
def load_embeddings():
    embeddings = np.load(EMBEDDINGS_PATH)
    labels = np.load(LABELS_PATH)
    return embeddings, labels

model = load_face_model()
embeddings_db, labels_db = load_embeddings()

# -----------------------
# HELPER FUNCTIONS
# -----------------------
def preprocess_image(img: Image.Image) -> np.ndarray:
    """Convert uploaded image to model input."""
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0).astype(np.float32)

def get_embedding(img_array: np.ndarray) -> np.ndarray:
    """Generate embedding from image array using trained model."""
    embedding = model.predict(img_array, verbose=0)
    return embedding

def recognize_face(img_array: np.ndarray, threshold: float = 0.6) -> str:
    """Compare embedding with DB and return closest match."""
    emb = get_embedding(img_array)
    dists = np.linalg.norm(embeddings_db - emb, axis=1)
    min_idx = np.argmin(dists)
    if dists[min_idx] < threshold:
        return labels_db[min_idx]
    else:
        return "Unknown"

# -----------------------
# STREAMLIT APP
# -----------------------
st.title("🧑 Face Recognition App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Show uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess + Predict
    img_array = preprocess_image(image)
    prediction = recognize_face(img_array)

    st.markdown(f"### ✅ Prediction: **{prediction}**")
