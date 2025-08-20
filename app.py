# app.py
import io
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mb_preprocess
from typing import Tuple

# -----------------------
# CONFIG / CONSTANTS
# -----------------------
IMG_SIZE = (160, 160)                       # must match training
MODEL_PATH = "best_model.h5"               # ensure this exists
CLASS_LABELS = ["Happy", "Sad", "Neutral", "Angry"]  # order must match training

# -----------------------
# CACHED RESOURCES
# -----------------------
@st.cache_resource
def get_model():
    try:
        # load without trying to restore optimizer state
        m = load_model(MODEL_PATH, compile=False)
        # compile with a fresh optimizer (match training loss/metrics)
        m.compile(optimizer=Adam(learning_rate=1e-4),
                  loss="categorical_crossentropy",   # or the loss you used
                  metrics=["accuracy"])
        return m
    except Exception as e:
        st.session_state.setdefault("_model_load_error", str(e))
        return None

@st.cache_resource
def get_haar_cascades() -> Tuple[cv2.CascadeClassifier, cv2.CascadeClassifier]:
    fc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    ec = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    return fc, ec

# -----------------------
# IMAGE PREPROCESSING (uses MobileNetV2 preprocessing)
# -----------------------
def preprocess_pil_image(pil_img: Image.Image) -> np.ndarray:
    """
    Convert PIL -> model input:
    - Convert to numpy, ensure 3 channels
    - Convert RGB -> BGR to match cv2.imread pipeline used in training
    - Resize to IMG_SIZE
    - Apply MobileNetV2 preprocess_input (scales to [-1,1])
    - Add batch dim
    """
    img_np = np.array(pil_img)  # PIL gives RGB
    if img_np.ndim == 2:  # grayscale -> 3-channel
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    # Convert RGB -> BGR (this keeps parity with cv2.imread used during training)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img_bgr, IMG_SIZE)
    # MobileNetV2 preprocess_input expects inputs in RGB convention normally,
    # but we apply the exact same pipeline used in training (cv2 -> mb_preprocess).
    img_pre = mb_preprocess(img_resized.astype("float32"))
    return np.expand_dims(img_pre, axis=0)

# -----------------------
# EYE STATUS DETECTION
# -----------------------
def check_eye_status_from_bytes(image_bytes: bytes) -> str:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return "Error: Could not decode image for eye detection."

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade, eye_cascade = get_haar_cascades()

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return "No face detected."

    for (x, y, w, h) in faces:
        eye_region_gray = gray[y : y + h // 2, x : x + w]
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

# Show model load error if any
model = get_model()
if model is None and "_model_load_error" in st.session_state:
    st.error(f"Failed to load model at '{MODEL_PATH}': {st.session_state['_model_load_error']}")
    st.stop()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file and model is not None:
    image_bytes = uploaded_file.read()
    st.image(image_bytes, caption="Uploaded Image", use_container_width=True)

    # Emotion prediction
    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        processed = preprocess_pil_image(pil_img)        # uses MobileNetV2 preprocessing
        preds = model.predict(processed)                 # shape (1, num_classes)
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
