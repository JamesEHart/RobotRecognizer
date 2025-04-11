from ultralytics import YOLO

# Load a pre-trained YOLOv8 model (smallest version)
model = YOLO('yolov8n.pt')

# Train the model using your data.yaml
model.train(
    data='Data.yaml',
    epochs=50,
    imgsz=640,
    batch=4,           # Change based on GPU/CPU capability
    name='my_yolo_model'
)

# Optional: Save the model path
model_path = model.ckpt_path if hasattr(model, 'ckpt_path') else 'runs/detect/my_yolo_model/weights/best.pt'

print(f"Model trained and saved to: {model_path}")
