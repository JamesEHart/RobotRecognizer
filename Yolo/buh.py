from ultralytics import YOLO

# Load your trained model
model = YOLO('./runs/detect/my_yolo_model8/weights/best.pt')

# Run inference on an image
results = model('image.png')  # Replace with your test image
results.show()                # Shows the image with boxes
