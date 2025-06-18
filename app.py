import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Define image dimensions (should be the same as used during training)
IMG_SIZE = (64, 64)  # This should match the image size used during training

# Load the trained model (make sure the model is in the same directory or provide the path)
model = load_model('cnn_model.h5')

# Function to preprocess the image
def preprocess_image(image):
    # Convert the uploaded image to an OpenCV format (BGR)
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Resize the image to the expected input size for the model
    img = cv2.resize(img, IMG_SIZE)
    
    # Normalize the image (scaling pixel values to [0, 1])
    img = img.astype('float32') / 255.0
    
    # Add an extra dimension to match the input shape (batch size dimension)
    img = np.expand_dims(img, axis=0)
    
    return img

# Streamlit app layout
st.title("Emotion Detection from Image")
st.write("Upload an image to predict its emotion.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    image = Image.open(uploaded_file)
    processed_image = preprocess_image(image)

    # Predict the class of the image
    predictions = model.predict(processed_image)

    # Get the predicted class index (highest probability)
    predicted_class_index = np.argmax(predictions)

    # Define the class labels (make sure they match the order used during training)
    class_labels = ['Happy', 'Sad', 'Neutral', 'Angry']  # Modify according to your classes
    predicted_class = class_labels[predicted_class_index]

    # Display the prediction result
    st.write(f"Predicted Class: {predicted_class} with probability: {predictions[0][predicted_class_index]:.2f}")
    
    
    
        
    import cv2
    
    def check_eye_status(image_path):
        # Load the image
        image = cv2.imread(image_path)
        
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
        # Load the pre-trained classifiers for face and eyes
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
        if len(faces) == 0:
            return "No face detected."
    
        # Loop through each detected face
        for (x, y, w, h) in faces:
            # Crop the face region
            face_region = gray[y:y+h, x:x+w]
    
            # Detect eyes in the face region
            eyes = eye_cascade.detectMultiScale(face_region)
    
            # Check if any eyes are detected
            if len(eyes) > 0:
                st.write("Eye is open.")

                return "Eye is open."

            else:
                st.write("Eye is Closed")

                return "Eye is closed."
            
        
        return "No eyes detected."
    
    # Example usage
    # Path to your image file
    image_path = "images/00003502.jpg"
    status = check_eye_status(image_path)
    print(status)

    
    
    
    
