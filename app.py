import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.models import load_model
from tensorflow.keras.mixed_precision import set_global_policy
from PIL import Image

# Set global policy — use 'float32' unless your model was trained with mixed precision
set_global_policy('float32')

# Monkey-patch InputLayer to handle 'batch_shape' in newer TF versions
_orig_init = InputLayer.__init__
def _patched_init(self, *args, **kwargs):
    if 'batch_shape' in kwargs:
        kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
    return _orig_init(self, *args, **kwargs)
InputLayer.__init__ = _patched_init

# Define image dimensions used during training
IMG_SIZE = (64, 64)

# Load the trained model
model = load_model('cnn_model.h5', compile=False)

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Streamlit app UI
st.title("Emotion Detection from Image")
st.write("Upload an image to predict its emotion.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    image = Image.open(uploaded_file)
    processed_image = preprocess_image(image)

    # Predict emotion
    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions)
    class_labels = ['Happy', 'Sad', 'Neutral', 'Angry']  # Adjust as needed
    predicted_class = class_labels[predicted_class_index]

    st.write(f"Predicted Class: {predicted_class} "
             f"with probability: {predictions[0][predicted_class_index]:.2f}")

# OPTIONAL: Eye detection function (for demo/testing)
def check_eye_status(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return "Invalid image path."

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return "No face detected."

    for (x, y, w, h) in faces:
        face_region = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_region)

        if len(eyes) > 0:
            st.write("Eye is open.")
            return "Eye is open."
        else:
            st.write("Eye is closed.")
            return "Eye is closed."

    return "No eyes detected."

# Example usage (if needed, or can comment out)
# image_path = "images/00003502.jpg"
# status = check_eye_status(image_path)
# print(status)
