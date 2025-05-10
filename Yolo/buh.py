from ultralytics import YOLO

# Load your trained model
model = YOLO('./runs/detect/my_yolo_model11/weights/best.pt')

# Run inference on an image
results = model('image3.png')  # Returns a list of Result objects

# Show the first (and only) result
results[0].save(filename='output4.png')