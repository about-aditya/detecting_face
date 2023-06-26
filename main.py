import cv2
import os

image_paths = ["00000.png", "00001.png", "00002.png", "00003.png"]

for path in image_paths:
    # Load the facial image
    image = cv2.imread(path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the pre-trained face cascade classifier
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # Iterate over each detected face
    for x, y, w, h in faces:
        # Define regions of interest (ROI) for eyes and mouth
        roi_eyes = gray[y : y + h, x : x + w]
        roi_mouth = gray[y + h // 2 : y + h, x : x + w]

        # Load the pre-trained eye cascade classifier
        eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )

        # Detect eyes in the eyes ROI
        eyes = eye_cascade.detectMultiScale(roi_eyes)

        # Iterate over each detected eye
        for ex, ey, ew, eh in eyes:
            # Draw a rectangle around the eye (relative to the face)
            cv2.rectangle(
                image, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 2
            )

        # Load the pre-trained mouth cascade classifier
        mouth_cascade = cv2.CascadeClassifier("haarcascade_mcs_mouth.xml")

        # Detect mouth in the mouth ROI
        mouth = mouth_cascade.detectMultiScale(roi_mouth)

        # Iterate over each detected mouth
        for mx, my, mw, mh in mouth:
            # Draw a rectangle around the mouth (relative to the face)
            cv2.rectangle(
                image,
                (x + mx, y + h // 2 + my),
                (x + mx + mw, y + h // 2 + my + mh),
                (0, 0, 255),
                2,
            )

    filename = os.path.splitext(os.path.basename(path))[0]
    extension = os.path.splitext(path)[1]

    # Set the output file path
    output_path = os.path.join(os.path.dirname(path), f"{filename}_detected{extension}")

    # Save and show the output image
    cv2.imwrite(output_path, image)
    cv2.imshow("Facial Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
