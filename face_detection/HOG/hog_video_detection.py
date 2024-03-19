import cv2
import dlib
import argparse

def hog_face_detection_video(video_path, output_path='./hog_face_detection_video.avi'):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video file")
        return

    # initialize hog + svm based face detector
    hog_face_detector = dlib.get_frontal_face_detector()

    # Get video properties
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = cap.get(5)

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # apply face detection (hog)
        faces_hog = hog_face_detector(frame, 1)

        # loop over detected faces
        for face in faces_hog:
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y

            # draw box over face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)

        # save output frame to video
        out.write(frame)

        # display output frame
        cv2.imshow("HOG Face Detection with dlib", frame)

        # exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release the video capture object, VideoWriter, and close all windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video with HOG face detection saved to: {output_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-v', '--video', default='Walk_Trim.mp4', help='path to video file')
    args = ap.parse_args()

    hog_face_detection_video(args.video)
