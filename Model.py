import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

# -------------------- Config --------------------
IMG_SIZE = (64, 64)
IMAGE_FOLDER = "/content/drive/MyDrive/images"
NUM_CLASSES = 4
CLASS_LABELS = ['Happy', 'Sad', 'Neutral', 'Angry']  # modify as needed
MODEL_PATH = "cnn_model.h5"
# ------------------------------------------------


# -------------------- Data Loader --------------------
def load_dataset(folder, num_classes):
    images, labels = [], []
    # Example assumes equal number of images per class, sequential naming
    images_per_class = len(os.listdir(folder)) // num_classes
    label_sequence = sum([[i] * images_per_class for i in range(num_classes)], [])

    for img_name, label in zip(sorted(os.listdir(folder)), label_sequence):
        if img_name.lower().endswith((".jpg", ".png")):
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, IMG_SIZE).astype("float32") / 255.0
            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels)


# -------------------- Model Builder --------------------
def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# -------------------- Training --------------------

images, labels = load_dataset(IMAGE_FOLDER, NUM_CLASSES)
X_train, X_val, y_train, y_val = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

model = build_model((*IMG_SIZE, 3), NUM_CLASSES)
model.fit(X_train, y_train, epochs=10, batch_size=32,
          validation_data=(X_val, y_val))
model.save(MODEL_PATH)

loss, acc = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {acc * 100:.2f}%")



# -------------------- Prediction --------------------
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, IMG_SIZE).astype("float32") / 255.0
    return np.expand_dims(img, axis=0)


def predict_image(model, image_path, class_labels):
    img = preprocess_image(image_path)
    preds = model.predict(img)
    idx = np.argmax(preds)
    return class_labels[idx], preds[0][idx]


# Example: predict on one image
test_image = os.path.join(IMAGE_FOLDER, "00000004.jpg")
pred_class, prob = predict_image(model, test_image, CLASS_LABELS)
print(f"Predicted Class: {pred_class} ({prob:.2f} confidence)")
