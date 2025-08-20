import io
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from typing import Tuple

# -----------------------
# CONFIG / CONSTANTS
# -----------------------
IMG_SIZE = (64, 64)                       # same as training
MODEL_PATH = "cnn_model.h5"               # ensure this exists
CLASS_LABELS = ["Happy", "Sad", "Neutral", "Angry"]  # keep order same as training

# -----------------------
# CACHED RESOURCES
# -----------------------
@st.cache_resource
def get_model():
    try:
        return load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load model at '{MODEL_PATH}': {e}")
        return None

@st.cache_resource
def get_haar_cascades() -> Tuple[cv2.CascadeClassifier, cv2.CascadeClassifier]:
    fc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    ec = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    return fc, ec

# -----------------------
# IMAGE PREPROCESSING
# -----------------------
def preprocess_pil_image(pil_img: Image.Image) -> np.ndarray:
    """
    Convert PIL image -> model input:
    - RGB -> BGR (preserve your original pipeline)
    - resize to IMG_SIZE, scale to [0,1], add batch dim
    """
    img_np = np.array(pil_img)                  # RGB (H,W,3)
    if img_np.ndim == 2:                        # grayscale -> convert to 3-channel
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    # Convert RGB -> BGR for OpenCV-style if that matches training pipeline
    try:
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    except Exception:
        img_bgr = img_np  # fallback if conversion fails
    img_resized = cv2.resize(img_bgr, IMG_SIZE)
    img_normalized = img_resized.astype("float32") / 255.0
    return np.expand_dims(img_normalized, axis=0)

# -----------------------
# EYE STATUS DETECTION
# -----------------------
def check_eye_status_from_bytes(image_bytes: bytes) -> str:
    """
    Uses OpenCV Haar cascades on an in-memory image (no temp files).
    Returns "Eye is open.", "Eye is closed.", or an error / no-face message.
    """
    # decode bytes into cv2 image (BGR)
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return "Error: Could not decode image for eye detection."

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade, eye_cascade = get_haar_cascades()

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return "No face detected."

    # default - assume eyes closed unless we detect eyes
    for (x, y, w, h) in faces:
        eye_region_gray = gray[y : y + h // 2, x : x + w]      # upper half of face
        eyes = eye_cascade.detectMultiScale(
            eye_region_gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(30, 30),
            maxSize=(80, 80),
        )
        if len(eyes) > 0:
            return "Eye is open."

    return "Eye is closed."

# -----------------------
# STREAMLIT UI
# -----------------------
st.title("Emotion + Eye Status Detection")
st.write("Upload an image to predict emotion and check eye status.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

model = get_model()  # cached

if uploaded_file and model is not None:
    # Read bytes once and reuse
    image_bytes = uploaded_file.read()

    # Display uploaded image
    st.image(image_bytes, caption="Uploaded Image", use_container_width=True)

    # Emotion prediction
    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        processed = preprocess_pil_image(pil_img)
        preds = model.predict(processed)          # shape (1, num_classes)
        pred_idx = int(np.argmax(preds[0]))
        pred_prob = float(preds[0][pred_idx])
        pred_label = CLASS_LABELS[pred_idx] if pred_idx < len(CLASS_LABELS) else f"Class {pred_idx}"
        st.subheader("Emotion Prediction")
        st.write(f"Predicted: **{pred_label}** — probability: `{pred_prob:.2f}`")
    except Exception as e:
        st.error(f"Emotion prediction failed: {e}")

    # Eye status
    try:
        eye_status = check_eye_status_from_bytes(image_bytes)
        st.subheader("Eye Status Check")
        st.write(f"Eye Status: **{eye_status}**")
    except Exception as e:
        st.error(f"Eye detection failed: {e}")

elif uploaded_file and model is None:
    st.error("Model could not be loaded. Check logs and model path.")
