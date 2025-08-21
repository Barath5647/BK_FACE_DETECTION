import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

st.title("YOLO Face Detection (Fine-Tuned)")

# Load fine-tuned model
model = YOLO("/content/drive/MyDrive/face_yolo_best.pt")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    # Convert uploaded image to numpy
    image = Image.open(uploaded).convert("RGB")
    img_array = np.array(image)

    # Run YOLO detection
    results = model.predict(img_array, conf=0.4)
    boxes = results[0].boxes.xyxy.cpu().numpy()   # [x0,y0,x1,y1]
    confs = results[0].boxes.conf.cpu().numpy()

    # Draw bounding boxes
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    for (x0,y0,x1,y1), conf in zip(boxes, confs):
        cv2.rectangle(img_bgr, (int(x0),int(y0)), (int(x1),int(y1)), (0,255,0), 2)
        cv2.putText(img_bgr, f"Face {conf:.2f}", (int(x0), int(y0)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
             caption="Detected Faces", use_column_width=True)
