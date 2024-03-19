# Import necessary libraries
import cv2
import random
import sys
import numpy as np

def generator():
    N = 10
    n = [random.randint(0, sys.maxsize) for _ in range(N)]
    b = str(n[0])
    a = b[:8]
    return a

# Open video file
cap = cv2.VideoCapture("G:/Git-projects/Object_Detection_project/vehicle_detection_Haar Cascade Classifier/input_01.mp4")

# Load car cascade classifier
car_cascade = cv2.CascadeClassifier('G:\\Git-projects\\Object_Detection_project\\vehicle_detection_Haar Cascade Classifier\\cars.xml')

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
output_video_path = "output_video_car_detection.mp4"
video_writer = cv2.VideoWriter(output_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while True:
    # Read each frame from the video
    ret, frames = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    # Detect cars in the frame
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=9)

    # Process each detected car
    for (x, y, w, h) in cars:
        plate = frames[y:y + h, x:x + w]
        cv2.rectangle(frames, (x, y), (x + w, y + h), (51, 51, 255), thickness=5)
        cv2.putText(frames, 'Car', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 5)
        cv2.imshow('car', plate)
        
        # Save the frame with the detected car
        cv2.imwrite(f"G:\\Git-projects\\Object_Detection_project\\vehicle_detection_Haar Cascade Classifier\\save\\{generator()}.jpg", frames)

    # Write the frame to the output video
    video_writer.write(frames)

    # Display the processed frame
    frames = cv2.resize(frames, (600, 400))
    cv2.imshow('Car Detection System', frames)

    # Wait for the user to press 'Esc' key to exit
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Release video capture, video writer, and close all windows
cap.release()
video_writer.release()
cv2.destroyAllWindows()
