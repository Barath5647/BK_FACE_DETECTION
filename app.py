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
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_file, custom_object_scope, register_keras_serializable
from tensorflow.keras.mixed_precision import Policy as DTypePolicy
from keras.saving import object_registration  # <-- new
from PIL import Image

# --- 1) Safe registration ---
if "Custom>DTypePolicy" not in object_registration._GLOBAL_CUSTOM_OBJECTS:
    register_keras_serializable(package="Custom", name="DTypePolicy")(DTypePolicy)

# --- 2) Monkey-patch InputLayer for TF 2.12+ ---
_orig_init = InputLayer.__init__
def _patched_init(self, *args, **kwargs):
    if "batch_shape" in kwargs:
        kwargs["batch_input_shape"] = kwargs.pop("batch_shape")
    return _orig_init(self, *args, **kwargs)
InputLayer.__init__ = _patched_init

# --- 3) Load or download model ---
MODEL_FILENAME = "cnn_model.h5"
MODEL_URL = (
    "https://raw.githubusercontent.com/"
    "Barath5647/BK_FACE_DETECTION/main/cnn_model.h5"
)

with custom_object_scope({"DTypePolicy": DTypePolicy}):
    try:
        model = load_model(MODEL_FILENAME, compile=False)
    except (OSError, IOError):
        path = get_file(MODEL_FILENAME, MODEL_URL, cache_subdir=".")
        model = load_model(path, compile=False)

# --- 4) Preprocessing ---
IMG_SIZE = (64, 64)
def preprocess_image(img: Image.Image) -> np.ndarray:
    arr = np.array(img)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    resized = cv2.resize(bgr, IMG_SIZE)
    normed = resized.astype("float32") / 255.0
    return np.expand_dims(normed, axis=0)

# --- 5) Streamlit UI ---
st.title("Emotion Detection from Image")
st.write("Upload an image to predict its emotion and check eye status.")

uploaded = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded:
    st.image(uploaded, caption="Uploaded Image", use_container_width=True)
    img = Image.open(uploaded)
    batch = preprocess_image(img)

    # Predict emotion
    preds = model.predict(batch)[0]
    idx  = np.argmax(preds)
    labels = ["Happy", "Sad", "Neutral", "Angry"]
    st.write(f"**Emotion:** {labels[idx]} ({preds[idx]:.2f} confidence)")

    # Eye status
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    face_c = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_c  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    faces = face_c.detectMultiScale(gray, 1.1, 5)

    if len(faces) == 0:
        st.write("No face detected.")
    else:
        x, y, w, h = faces[0]
        eyes = eye_c.detectMultiScale(gray[y:y+h, x:x+w])
        st.write("Eye is open." if len(eyes) > 0 else "Eye is closed.")

