import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import tempfile
import os

# --- Configuration ---
# Define image dimensions (should be the same as used during training)
IMG_SIZE = (64, 64)

# --- Model Loading ---
# Load the trained model.
# IMPORTANT: Ensure 'cnn_model.h5' is uploaded to your Colab environment
# or provide the correct path to it.
try:
    model = load_model('cnn_model.h5')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("Please ensure 'cnn_model.h5' is uploaded to your Colab environment.")
    st.stop() # Stop the Streamlit app if model isn't found

# --- Image Preprocessing Function ---
def preprocess_image(image_pil):
    """
    Preprocesses a PIL Image for model prediction.
    Converts to OpenCV format, resizes, normalizes, and adds batch dimension.
    """
    # Convert the uploaded PIL image to a NumPy array (RGB)
    img_np = np.array(image_pil)

    # Convert RGB to BGR for OpenCV compatibility
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Resize the image to the expected input size for the model
    img_resized = cv2.resize(img_bgr, IMG_SIZE)

    # Normalize the image (scaling pixel values to [0, 1])
    img_normalized = img_resized.astype('float32') / 255.0

    # Add an extra dimension to match the input shape (batch size dimension)
    img_expanded = np.expand_dims(img_normalized, axis=0)

    return img_expanded

# --- Eye Status Detection Function ---
def check_eye_status(image_file_object):
    """
    Checks if eyes are open or closed in an uploaded image using OpenCV Haar Cascades.
    Returns a string indicating the status.
    """
    # Save the uploaded file to a temporary location for OpenCV to read
    # Use delete=False to ensure the file exists for cv2.imread before deletion
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(image_file_object.getvalue()) # Use getvalue() for BytesIO object
        temp_image_path = tmp.name

    try:
        # Load the image using OpenCV
        image = cv2.imread(temp_image_path)
        if image is None:
            return "Error: Could not load image for eye detection."

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Load the pre-trained classifiers for face and eyes
        # These are usually included with OpenCV installations
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        eye_status_message = "No face detected." # Default message

        if len(faces) == 0:
            eye_status_message = "No face detected."
        else:
            # Initial assumption: if a face is found but no eyes are detected yet,
            # we'll assume "Eye is closed" or detection failed.
            eye_status_message = "Eye is closed."

            # Loop through each detected face
            for (x, y, w, h) in faces:
                # Crop the face region (often eyes are in the upper half of the face)
                # Let's focus eye detection on the upper half of the detected face
                eye_region_gray = gray[y : y + h // 2, x : x + w]

                # Detect eyes in the eye region
                # Adjusted parameters for stricter detection to reduce false positives
                eyes = eye_cascade.detectMultiScale(
                    eye_region_gray,
                    scaleFactor=1.1,
                    minNeighbors=6, # Increased minNeighbors
                    minSize=(30, 30), # Increased minSize
                    maxSize=(80, 80) # Added maxSize
                )

                if len(eyes) > 0:
                    eye_status_message = "Eye is open."
                    break # If any open eyes are found, we can assume the person's eyes are open
                # If no eyes are found in this face region, the default 'Eye is closed' remains
                # or will be set by the next face if it also has no detected eyes.

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

    return eye_status_message

# --- Streamlit App Layout ---
st.title("Emotion and Eye Status Detection from Image")
st.write("Upload an image to predict its emotion and check eye status.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    # Changed 'use_column_width' to 'use_container_width' to address deprecation warning
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # --- Emotion Detection ---
    image_pil = Image.open(uploaded_file)
    processed_image = preprocess_image(image_pil)

    # Predict the class of the image
    predictions = model.predict(processed_image)

    # Get the predicted class index (highest probability)
    predicted_class_index = np.argmax(predictions)

    # Define the class labels (make sure they match the order used during training)
    class_labels = ['Happy', 'Sad', 'Neutral', 'Angry'] # Modify according to your classes
    predicted_class = class_labels[predicted_class_index]

    # Display the prediction result
    st.subheader("Emotion Prediction:")
    st.write(f"Predicted Emotion: **{predicted_class}** with probability: `{predictions[0][predicted_class_index]:.2f}`")

    # --- Eye Status Check ---
    st.subheader("Eye Status Check:")
    # Reset file pointer to the beginning before passing to check_eye_status
    uploaded_file.seek(0)
    eye_status = check_eye_status(uploaded_file)
    st.write(f"Eye Status: **{eye_status}**")
