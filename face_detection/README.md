# HOG-Based Face Detection with dlib
## Overview
This repository contains a Python script for face detection using the dlib library with the Histogram of Oriented Gradients (HOG) technique. The script is designed to work with both images and videos, providing a flexible solution for face detection tasks.

## Features
HOG-Based Face Detection:

 Utilizes the HOG + SVM face detection algorithm provided by dlib.

**Image Detection:** 
Detect faces in images and save the results.
![output1](HOG/output/hog_face_detection_image.png)
![output3](HOG/output/hog_face_detection_image3.png)
![output5](HOG/output/hog_face_detection_image5.png)
![output6](HOG/output/hog_face_detection_image6.png)

the output of blur image. very well!

![output2](HOG/output/hog_face_detection_image2.png)
![output4](HOG/output/hog_face_detection_image4.png)


**Video Detection**:Perform real-time face detection in videos and save the processed video.

[output video file](HOG/output/hog_face_detection_video.mp4)

![output video file](HOG/output/videogif.gif)


## Requirements
    Python 3.x
    OpenCV
    dlib
    (Additional dependencies as needed)

## Usage
**HOG-Based Face Detection for Image**

    bash
    Copy code
    python hog_image_detection.py -i path/to/your/image.jpg

**HOG-Based Face Detection for Video**

    bash
    Copy code
    python hog_video_detection.py -v path/to/your/video.mp4

**Parameters**

    -i, --image: Path to the input image file.
    -v, --video: Path to the input video file.

**Output**

Detected faces will be highlighted with bounding boxes.
Processed images and videos will be saved in the current working directory.

**Dependencies**

    OpenCV
    dlib
    Notes:
    Make sure to install the required dependencies before running the scripts.
    Adjust file paths and names as needed.
