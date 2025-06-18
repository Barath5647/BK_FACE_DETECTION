# import os
# import streamlit as st
# import cv2
# import numpy as np
# from tensorflow.keras.layers import InputLayer
# from tensorflow.keras.models import load_model
# from tensorflow.keras.mixed_precision import Policy
# from tensorflow.keras.utils import custom_object_scope
# from PIL import Image

# # --- Monkey‑patch InputLayer to accept `batch_shape` in TF 2.12+ ---
# _orig_init = InputLayer.__init__
# def _patched_init(self, *args, **kwargs):
#     if "batch_shape" in kwargs:
#         kwargs["batch_input_shape"] = kwargs.pop("batch_shape")
#     return _orig_init(self, *args, **kwargs)
# InputLayer.__init__ = _patched_init

# IMG_SIZE = (64, 64)
# MODEL_PATH = "cnn_model.h5"

# # Check for model file
# if not os.path.exists(MODEL_PATH):
#     st.error(f"❌ Model file not found at `{MODEL_PATH}`. Please upload it to the app directory.")
#     st.stop()

# # Load model with DTypePolicy in scope
# with custom_object_scope({"DTypePolicy": Policy}):
#     model = load_model(MODEL_PATH, compile=False)

# def preprocess_image(img: Image.Image) -> np.ndarray:
#     arr = np.array(img)
#     bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
#     resized = cv2.resize(bgr, IMG_SIZE)
#     normed = resized.astype("float32") / 255.0
#     return np.expand_dims(normed, axis=0)

# def eye_status(bgr_img: np.ndarray) -> str:
#     gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
#     fc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
#     ec = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
#     faces = fc.detectMultiScale(gray, 1.1, 5)
#     if not len(faces):
#         return "No face detected."
#     for (x, y, w, h) in faces:
#         roi = gray[y : y + h, x : x + w]
#         eyes = ec.detectMultiScale(roi)
#         return "Eye open." if len(eyes) else "Eye closed."
#     return "No eyes found."

# st.title("Emotion Detection from Image")
# st.write("Upload a .jpg or .png and get its emotion + eye status.")

# uploaded = st.file_uploader("Choose an image...", type=["jpg", "png"])
# if uploaded:
#     img = Image.open(uploaded)
#     st.image(img, use_container_width=True, caption="Your uploaded image")

#     # Predict emotion
#     batch = preprocess_image(img)
#     preds = model.predict(batch)[0]
#     idx = int(np.argmax(preds))
#     labels = ["Happy", "Sad", "Neutral", "Angry"]
#     st.write(f"**Predicted emotion:** {labels[idx]} — {preds[idx]:.2f}")

#     # Eye status
#     bgr_arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
#     st.write(f"**Eye status:** {eye_status(bgr_arr)}")


# import os
# import streamlit as st
# import cv2
# import numpy as np
# import requests
# from tensorflow.keras.layers import InputLayer
# from tensorflow.keras.models import load_model
# from tensorflow.keras.mixed_precision import Policy
# from tensorflow.keras.utils import custom_object_scope
# from PIL import Image

# # --- Monkey‑patch InputLayer to accept `batch_shape` in TF 2.12+ ---
# _orig_init = InputLayer.__init__
# def _patched_init(self, *args, **kwargs):
#     if "batch_shape" in kwargs:
#         kwargs["batch_input_shape"] = kwargs.pop("batch_shape")
#     return _orig_init(self, *args, **kwargs)
# InputLayer.__init__ = _patched_init

# # Settings
# IMG_SIZE   = (64, 64)
# MODEL_FILE = "cnn_model.h5"
# RAW_URL    = (
#     "https://raw.githubusercontent.com/"
#     "Barath5647/BK_FACE_DETECTION/"
#     "c5df19c08d70a6638cc20c4db6eb3b0a37410938/"
#     "cnn_model.h5"
# )

# # If model is missing, download it
# if not os.path.exists(MODEL_FILE):
#     st.warning(f"`{MODEL_FILE}` not found locally. Downloading from GitHub…")
#     try:
#         resp = requests.get(RAW_URL, stream=True)
#         resp.raise_for_status()
#         with open(MODEL_FILE, "wb") as f:
#             for chunk in resp.iter_content(1 << 20):
#                 f.write(chunk)
#         st.success("Model downloaded successfully!")
#     except Exception as e:
#         st.error(f"Failed to download model:\n{e}")
#         st.stop()

# # Load the model with mixed‑precision policy in scope
# with custom_object_scope({"DTypePolicy": Policy}):
#     model = load_model(MODEL_FILE, compile=False)

# def preprocess_image(img: Image.Image) -> np.ndarray:
#     arr = np.array(img)
#     bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
#     resized = cv2.resize(bgr, IMG_SIZE)
#     normed  = resized.astype("float32") / 255.0
#     return np.expand_dims(normed, axis=0)

# def eye_status(bgr_img: np.ndarray) -> str:
#     gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
#     fc   = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
#     ec   = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
#     faces = fc.detectMultiScale(gray, 1.1, 5)
#     if len(faces) == 0:
#         return "No face detected."
#     for (x, y, w, h) in faces:
#         roi  = gray[y : y + h, x : x + w]
#         eyes = ec.detectMultiScale(roi)
#         return "Eye open." if len(eyes) else "Eye closed."
#     return "No eyes found."

# st.title("Emotion & Eye‑Status Detection")
# st.write("Upload a JPG/PNG and get its emotion + eye status.")

# uploaded = st.file_uploader("Choose an image...", type=["jpg","png"])
# if uploaded:
#     img = Image.open(uploaded)
#     st.image(img, use_container_width=True, caption="Uploaded image")

#     # Emotion prediction
#     batch = preprocess_image(img)
#     preds = model.predict(batch)[0]
#     idx   = int(np.argmax(preds))
#     labels = ["Happy","Sad","Neutral","Angry"]
#     st.write(f"**Emotion:** {labels[idx]} — {preds[idx]:.2f}")

#     # Eye status
#     bgr_arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
#     st.write(f"**Eye status:** {eye_status(bgr_arr)}")

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from PIL import Image
import tempfile
import os

# Constants
IMG_SIZE = (64, 64)
MODEL_FILENAME = "cnn_model.h5"
MODEL_URL = "https://github.com/Barath5647/BK_FACE_DETECTION/raw/main/cnn_model.h5"

# Download model if not present
if not os.path.exists(MODEL_FILENAME):
    import tensorflow as tf
    MODEL_FILENAME = tf.keras.utils.get_file(MODEL_FILENAME, MODEL_URL, cache_subdir='.')

# Load model
model = load_model(MODEL_FILENAME, compile=False)

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# Eye landmarks
RIGHT_EYE = [33, 133, 160, 159, 158, 157, 173, 153]
LEFT_EYE  = [362, 263, 387, 386, 385, 384, 398, 382]

# Functions
def preprocess_image(image):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def eye_aspect_ratio(eye_points):
    # Simple vertical distance calculation for demo
    top = eye_points[2][1] + eye_points[3][1]
    bottom = eye_points[6][1] + eye_points[7][1]
    left = eye_points[0][0]
    right = eye_points[4][0]
    
    vertical = abs((top - bottom) / 2.0)
    horizontal = abs(right - left)
    
    if horizontal == 0:
        return 0
    
    ratio = vertical / horizontal
    return ratio

def detect_eye_status(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    
    if not results.multi_face_landmarks:
        return "No face detected"
    
    ih, iw, _ = image.shape
    for face_landmarks in results.multi_face_landmarks:
        right_eye_points = [(int(face_landmarks.landmark[idx].x * iw), 
                             int(face_landmarks.landmark[idx].y * ih)) for idx in RIGHT_EYE]
        left_eye_points  = [(int(face_landmarks.landmark[idx].x * iw), 
                             int(face_landmarks.landmark[idx].y * ih)) for idx in LEFT_EYE]
        
        right_ratio = eye_aspect_ratio(right_eye_points)
        left_ratio = eye_aspect_ratio(left_eye_points)
        
        threshold = 0.2  # You can tune this
        
        right_status = "Open" if right_ratio > threshold else "Closed"
        left_status  = "Open" if left_ratio > threshold else "Closed"
        
        return f"Right Eye: {right_status}, Left Eye: {left_status}"
    
    return "No eyes detected"

# --- Streamlit UI ---
st.title("Emotion Detection + Eye Status (Mediapipe FaceMesh)")
st.write("Upload an image to predict emotion and detect eye status.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Convert PIL to OpenCV
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Emotion prediction
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions)
    class_labels = ['Happy', 'Sad', 'Neutral', 'Angry']  # Adjust if needed
    predicted_class = class_labels[predicted_class_index]
    
    # Eye status
    eye_status = detect_eye_status(img_cv)
    
    # Results
    st.header("Results:")
    st.write(f"**Emotion:** {predicted_class} ({predictions[0][predicted_class_index]:.2f})")
    st.write(f"**Eye Status:** {eye_status}")
