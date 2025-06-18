import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.models import load_model
from tensorflow.keras.mixed_precision import Policy
from tensorflow.keras.utils import custom_object_scope
from PIL import Image

# --- Monkey‑patch InputLayer to accept `batch_shape` in TF 2.12+ ---
_orig_init = InputLayer.__init__
def _patched_init(self, *args, **kwargs):
    if "batch_shape" in kwargs:
        kwargs["batch_input_shape"] = kwargs.pop("batch_shape")
    return _orig_init(self, *args, **kwargs)
InputLayer.__init__ = _patched_init

# Image size used at training
IMG_SIZE = (64, 64)

# Load model inside a scope that knows "DTypePolicy" → Policy
with custom_object_scope({"DTypePolicy": Policy}):
    model = load_model("cnn_model.h5", compile=False)

def preprocess_image(img: Image.Image) -> np.ndarray:
    """Convert to BGR, resize, normalize, add batch axis."""
    arr = np.array(img)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    resized = cv2.resize(bgr, IMG_SIZE)
    normed = resized.astype("float32") / 255.0
    return np.expand_dims(normed, axis=0)

st.title("Emotion Detection from Image")
st.write("Upload a .jpg or .png to see its predicted emotion.")

uploaded = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Your image", use_column_width=True)

    batch = preprocess_image(img)
    preds = model.predict(batch)[0]
    idx = int(np.argmax(preds))
    labels = ["Happy", "Sad", "Neutral", "Angry"]
    st.write(f"**Predicted:** {labels[idx]}  — {preds[idx]:.2f}")

    # Optional: detect if eyes are open/closed
    def eye_status(bgr_img: np.ndarray) -> str:
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        fc = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        ec = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        faces = fc.detectMultiScale(gray, 1.1, 5)
        if len(faces) == 0:
            return "No face detected."
        for (x, y, w, h) in faces:
            roi = gray[y : y + h, x : x + w]
            eyes = ec.detectMultiScale(roi)
            return "Eye open." if len(eyes) > 0 else "Eye closed."
        return "No eyes found."

    bgr_arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    st.write(f"**Eye status:** {eye_status(bgr_arr)}")
