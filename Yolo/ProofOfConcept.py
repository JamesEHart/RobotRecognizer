from ultralytics import YOLO
import cv2
import numpy as np
import sys
sys.path.append('./sort')  # Add the sort folder to the Python path

from sort import Sort  # Now you can use the tracker

# === Load YOLO model ===
model = YOLO('./runs/detect/my_yolo_model11/weights/best.pt')

# === Load video ===
video_path = 'input_video.mp4'
cap = cv2.VideoCapture(video_path)

# === Get video properties ===
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# === Output video writer ===
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

# === Initialize trace image ===
trace_img = np.zeros((height, width, 3), dtype=np.uint8)

# === Initialize SORT tracker ===
tracker = Sort(max_age=10, min_hits=2, iou_threshold=0.3)

# === Robot path history ===
robot_paths = {}

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Stop after 30 seconds
    if frame_count > fps * 30:
        break

    if frame_count % 1 == 0:
        # === Run YOLO detection ===
        results = model(frame)[0]
        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            cls = int(box.cls[0])

            # Optional: filter by specific class if needed
            # if cls != your_robot_class_id:
            #     continue

            detections.append([x1.item(), y1.item(), x2.item(), y2.item(), conf.item()])

        # === Track detections ===
        detections_np = np.array(detections)
        tracks = tracker.update(detections_np)

        for track in tracks:
            x1, y1, x2, y2, track_id = track.astype(int)
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Robot {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Add center to robot's path
            if track_id not in robot_paths:
                robot_paths[track_id] = []
            robot_paths[track_id].append(center)

            # Draw trace line on trace image
            if len(robot_paths[track_id]) > 1:
                for i in range(1, len(robot_paths[track_id])):
                    pt1 = robot_paths[track_id][i - 1]
                    pt2 = robot_paths[track_id][i]
                    cv2.line(trace_img, pt1, pt2, (0, 0, 255), 2)

    # Write frame to output
    out.write(frame)
    frame_count += 1

# === Save final trace image ===
cv2.imwrite('robot_trace_all.png', trace_img)

# === Cleanup ===
cap.release()
out.release()
cv2.destroyAllWindows()
