from ultralytics import YOLO

# Load pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Train on our dataset
model.train(
    data="/content/drive/MyDrive/faces.yaml",   # dataset config
    epochs=50,           # adjust as needed
    imgsz=640,
    batch=16,
    project="/content/drive/MyDrive/yolo_runs", 
    name="face_yolo"
)

print("✅ Training finished, model saved in runs/detect/face_yolo/weights/best.pt")
