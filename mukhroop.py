# Importing libraries
import cv2

def detect_faces(image_path):
    # Loading the image
    image = cv2.imread(image_path)

    # Converting the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Loading the pre-trained face cascade from OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detecting faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)

    # Drawing rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Saving the output image with rectangles
    output_path = 'output.jpg'
    cv2.imwrite(output_path, image)
    
    return output_path

if __name__ == "__main__":
    # usage case
    image_path = 'images/test_image.jpg'
    output_path = detect_faces(image_path)
    print(f"Face detection completed. Result saved to {output_path}")
