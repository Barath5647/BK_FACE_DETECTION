# app.py (drop-in replacement)
import io
import os
import tempfile
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mb_preprocess
from typing import Tuple

# optional: used when fixing .h5
import h5py

# -----------------------
# CONFIG / CONSTANTS
# -----------------------
IMG_SIZE = (160, 160)
MODEL_PATH = "best_model.h5"
CLASS_LABELS = ["Happy", "Sad", "Neutral", "Angry"]

# -----------------------
# CACHED RESOURCES (robust model loader)
# -----------------------
@st.cache_resource
def get_model():
    """
    Tries multiple strategies to load the model:
      1) normal load_model(MODEL_PATH)
      2) load_model(..., compile=False) + recompile
      3) create a temporary cleaned HDF5 copy without optimizer_weights and load that (compile=False) + recompile
    Saves a human-friendly message to st.session_state['_model_load_error'] if anything odd happens.
    """
    st.session_state.pop("_model_load_error", None)  # clear previous
    # helper to compile consistently
    def compile_fresh(m):
        try:
            m.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
        except Exception:
            # if compile fails for any reason, ignore for inference
            pass
        return m

    # 1) try normal load
    try:
        m = load_model(MODEL_PATH)
        st.session_state["_model_load_error"] = "Loaded normally (with optimizer)."
        return m
    except Exception as e1:
        # record and try compile=False
        st.session_state["_model_load_error"] = f"Normal load failed: {e1!r}"
    # 2) try compile=False
    try:
        m = load_model(MODEL_PATH, compile=False)
        m = compile_fresh(m)
        st.session_state["_model_load_error"] += f" | Loaded with compile=False."
        return m
    except Exception as e2:
        st.session_state["_model_load_error"] += f" | load(..., compile=False) failed: {e2!r}"

    # 3) try creating a cleaned temporary copy without optimizer_weights
    try:
        if not os.path.exists(MODEL_PATH):
            st.session_state["_model_load_error"] += " | File not found."
            return None

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")
        tmp_path = tmp.name
        tmp.close()

        # copy everything except optimizer_weights
        with h5py.File(MODEL_PATH, "r") as src, h5py.File(tmp_path, "w") as dst:
            for k in src.keys():
                if k == "optimizer_weights":
                    # skip optimizer group that triggers dtype errors
                    continue
                src.copy(k, dst)
            # also copy root attrs if any
            for key, val in src.attrs.items():
                try:
                    dst.attrs[key] = val
                except Exception:
                    # ignore attributes that can't be copied
                    pass

        # try loading the cleaned file
        m = load_model(tmp_path, compile=False)
        m = compile_fresh(m)
        st.session_state["_model_load_error"] += " | Loaded from cleaned HDF5 copy (optimizer removed)."
        # remove the temp file
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return m
    except Exception as e3:
        st.session_state["_model_load_error"] += f" | cleaned-copy attempt failed: {e3!r}"
        # final fallback
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
    img_np = np.array(pil_img)
    if img_np.ndim == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    # we convert RGB -> BGR to keep parity with cv2 pipeline used while training
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img_bgr, IMG_SIZE)
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
        eyes = eye_cascade.detectMultiScale(eye_region_gray, scaleFactor=1.1, minNeighbors=6, minSize=(30,30), maxSize=(80,80))
        if len(eyes) > 0:
            return "Eye is open."
    return "Eye is closed."

# -----------------------
# STREAMLIT UI
# -----------------------
st.title("Emotion + Eye Status Detection")
st.write("Upload an image to predict emotion and check eye status.")

# show TF CPU info message is normal — nothing to fix
st.caption("Note: CPU optimization message from TensorFlow (informational).")

model = get_model()
if model is None:
    err = st.session_state.get("_model_load_error", "Unknown error while loading model.")
    st.error(f"Model not loaded. Details: {err}")
    st.stop()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image_bytes = uploaded_file.read()
    st.image(image_bytes, caption="Uploaded Image", use_container_width=True)

    # Emotion prediction
    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        processed = preprocess_pil_image(pil_img)
        preds = model.predict(processed)
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
