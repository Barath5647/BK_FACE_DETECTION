import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.models import load_model
from tensorflow.keras.mixed_precision import Policy as DTypePolicy
from tensorflow.keras.utils import custom_object_scope, register_keras_serializable
from PIL import Image

# --- Register DTypePolicy so Keras can (de)serialize it properly ---
register_keras_serializable(package='Custom', name='DTypePolicy')(DTypePolicy)

# --- Monkey‑patch InputLayer to accept 'batch_shape' argument in TF 2.12+ ---
_orig_init = InputLayer.__init__
def _patched_init(self, *args, **kwargs):
    if 'batch_shape' in kwargs:
        kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
    return _orig_init(self, *args, **kwargs)
InputLayer.__init__ = _patched_init

# Define image dimensions (must match what you used during training)
IMG_SIZE = (64, 64)

# Load the trained model, skipping optimizer state
with custom_object_scope({'DTypePolicy': DTypePolicy}):
    model = load_model('cnn_model.h5', compile=False)

# Function to preprocess the uploaded image for your CNN
def preprocess_image(image: Image.Image) -> np.ndarray:
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)

# Streamlit UI
st.title("Emotion Detection from Image")
st.write("Upload an image to predict its emotion.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    image = Image.open(uploaded_file)
    processed = preprocess_image(image)

    preds = model.predict(processed)
    idx = np.argmax(preds[0])
    labels = ['Happy', 'Sad', 'Neutral', 'Angry']  # adjust as needed
    st.write(f"**Predicted:** {labels[idx]}  –  probability {preds[0][idx]:.2f}")

    # --- Optional: Eye‑status check ---
    def check_eye_status(img_array: np.ndarray) -> str:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        if len(faces) == 0:
            return "No face detected."
        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi)
            return "Eye is open." if len(eyes) > 0 else "Eye is closed."
        return "No eyes detected."

    # Convert back to BGR array for eye check
    bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    status = check_eye_status(bgr)
    st.write(f"**Eye Status:** {status}")
