from ultralytics import YOLO

# Load your trained model
model = YOLO('./runs/detect/my_yolo_model10/weights/best.pt')

# Run inference on an image
results = model('image.png')  # Returns a list of Result objects

# Show the first (and only) result
results[0].save(filename='output.png')