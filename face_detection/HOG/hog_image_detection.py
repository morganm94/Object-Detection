import cv2
import dlib
import argparse

def hog_face_detection_image(image_path, output_path='output/hog_face_detection_image6.png'):
    # load input image
    image = cv2.imread(image_path)
    # image = cv2.resize(image,(1380,720)) # hd or normal image  
    image = cv2.resize(image,(680,480)) # blur face image

    if image is None:
        print("Could not read input image")
        return

    # initialize hog + svm based face detector
    hog_face_detector = dlib.get_frontal_face_detector()

    # apply face detection (hog)
    faces_hog = hog_face_detector(image, 2)

    # loop over detected faces
    for face in faces_hog:
        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y

        # draw box over face
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 255), 2)

    # save output image
    cv2.imwrite(output_path, image)
    cv2.imshow("output",image)
    cv2.waitKey(0)
    print(f"Image with HOG face detection saved to: {output_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', default='image/img_6.jpg', help='path to image file')
    args = ap.parse_args()

    hog_face_detection_image(args.image)
