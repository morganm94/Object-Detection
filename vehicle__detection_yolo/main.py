# Import necessary libraries
import cv2
import os
import numpy as np
import torch
import pandas as pd
from ultralytics import YOLO

# Load the YOLOv8 model
yolo_model = YOLO("yolov8x.pt")

# Map numeric class labels to text labels
dict_classes = yolo_model.model.names

# Set confidence threshold
conf_threshold = 0.5

# Define specific class IDs for detection (e.g., car, truck)
class_IDS = [2, 3, 5, 7]

# Open video file
video_capture = cv2.VideoCapture("input_01.mp4")

# Create output folder if not exists
output_folder = "output_frames"
os.makedirs(output_folder, exist_ok=True)

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
output_video_path = "output_video.mp4"
video_writer = cv2.VideoWriter(output_video_path, fourcc, 20.0, (int(video_capture.get(3)), int(video_capture.get(4))))

# Loop through each frame in the video
while True:
    ret, frame = video_capture.read()
    
    if not ret:
        break

    # Perform object detection using YOLOv8
    y_hat = yolo_model.predict(frame, conf=conf_threshold, classes=class_IDS, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), verbose=False)
    
    # Extract bounding box information from predictions
    boxes = y_hat[0].boxes.xyxy.cpu().numpy()
    confs = y_hat[0].boxes.conf.cpu().numpy()
    class_ids = y_hat[0].boxes.cls.cpu().numpy()
    positions_frame = pd.DataFrame(y_hat[0].cpu().numpy().boxes.boxes, columns=['xmin', 'ymin', 'xmax', 'ymax', 'conf', 'class'])
    
    # Translate numeric class labels to text
    labels = [dict_classes[i] for i in class_ids]

    # For each detected object, draw bounding box and label
    for ix, row in enumerate(positions_frame.iterrows()):
        xmin, ymin, xmax, ymax, confidence, category = row[1].astype('int')

        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 1)
        cv2.putText(frame, str(labels[ix]), (int(xmin), int(ymin-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display a cropped version of the detected object
        cropped_img = frame[int(ymin):int(ymax), int(xmin):int(xmax)]
        cv2.imshow("crop_object", cv2.resize(cropped_img, (250, 250))) 

    # Save the frame to the output folder
    frame_number = int(video_capture.get(1))
    output_path = os.path.join(output_folder, f"frame_{frame_number}.jpg")
    cv2.imwrite(output_path, frame)

    # Write the frame to the output video
    video_writer.write(frame)

    # Display the original frame
    cv2.imshow("original", frame)
    
    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release video capture, video writer, and close all windows
video_capture.release()
video_writer.release()
cv2.destroyAllWindows()
