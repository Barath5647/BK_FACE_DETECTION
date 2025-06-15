import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Define image dimensions (resize all images to the same size)
IMG_SIZE = (64, 64)  # You can choose different sizes based on your dataset

# Define the path to the image folder
image_folder = 'images'

# Initialize the image data and labels lists
images = []
labels = []

# Example: 4 classes, with 5 images per class
num_classes = 4
images_per_class = 5
total_images = num_classes * images_per_class

# Manually define labels (or load from a CSV file if available)
# Assuming labels are in a sequence where each class gets its label number
labels_sequence = [0] * images_per_class + [1] * images_per_class + [2] * images_per_class + [3] * images_per_class

# Loop through the images in the folder
for img_name, label in zip(sorted(os.listdir(image_folder)), labels_sequence):
    img_path = os.path.join(image_folder, img_name)
    
    if img_name.endswith('.jpg') or img_name.endswith('.png'):
        # Read the image
        img = cv2.imread(img_path)
        img = cv2.resize(img, IMG_SIZE)  # Resize to the fixed size
        img = img.astype('float32') / 255.0  # Normalize the image
        
        # Append image and its label
        images.append(img)
        labels.append(label)

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Create a CNN model
model = Sequential()

# Add convolutional and pooling layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Flatten the output from the convolutional layers
model.add(Flatten())

# Add fully connected layers
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))  # Output layer (softmax for classification)

# Compile the model
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32)

# Save the model after training
model.save('cnn_model.h5')

# Evaluate the model on the validation set
loss, accuracy = model.evaluate(X_train, y_train)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")




import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Define image dimensions (should be the same as used during training)
IMG_SIZE = (64, 64)  # This should match the image size used during training

# Load the trained model
model = load_model('cnn_model.h5')

# Function to preprocess the image
def preprocess_image(image_path):
    # Read the image
    img = cv2.imread(image_path)
    
    # Resize the image to the expected input size for the model
    img = cv2.resize(img, IMG_SIZE)
    
    # Normalize the image (scaling pixel values to [0, 1])
    img = img.astype('float32') / 255.0
    
    # Add an extra dimension to match the input shape (batch size dimension)
    img = np.expand_dims(img, axis=0)
    
    return img

# Path to the single image you want to predict
image_path = 'images/00000004.jpg'

# Preprocess the image
processed_image = preprocess_image(image_path)

# Predict the class of the image
predictions = model.predict(processed_image)

# Get the predicted class index (highest probability)
predicted_class_index = np.argmax(predictions)

# Assuming the labels were [0, 1, 2, 3], you can map the class index back to the class label
class_labels = ['Happy', 'Sad', 'Neutral', 'Angry']  # Modify according to your classes
predicted_class = class_labels[predicted_class_index]

# Print the predicted class
print(f"Predicted Class: {predicted_class} with probability: {predictions[0][predicted_class_index]:.2f}")




















