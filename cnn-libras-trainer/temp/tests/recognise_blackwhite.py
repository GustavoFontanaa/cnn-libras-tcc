import cv2
import numpy as np
from keras.models import load_model

IMAGE_X, IMAGE_Y = 64, 64
CLASSES = 21
LETTERS = {
    '0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F', '6': 'G',
    '7': 'G', '8': 'I', '9': 'L', '10': 'M', '11': 'N', '12': 'O', '13': 'P',
    '14': 'Q', '15': 'R', '16': 'S', '17': 'T', '18': 'U', '19': 'V', '20': 'W', '21': 'Y'
}

MODEL_PATH = '../../models/cnn_model_LIBRAS_20200531_0304.h5'
classifier = load_model(MODEL_PATH)

def nothing(x):
    pass

def predictor(test_image):
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)

    class_index = np.argmax(result[0])  # Find the class with the highest probability
    return result, LETTERS[str(class_index)]

def configure_trackbars():
    """Create trackbars for adjusting HSV range."""
    cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
    cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
    cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

def get_trackbar_values():
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    return l_h, l_s, l_v, u_h, u_s, u_v

def main():
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("Trackbars")
    configure_trackbars()

    img_text = ['', '']

    while True:
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)

        # Get trackbar values
        l_h, l_s, l_v, u_h, u_s, u_v = get_trackbar_values()

        # Define the region for cropping
        img = cv2.rectangle(frame, (425, 100), (625, 300), (0, 255, 0), thickness=2)
        imcrop = img[102:298, 427:623]

        # Convert to HSV and apply the mask
        hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([l_h, l_s, l_v])
        upper_blue = np.array([u_h, u_s, u_v])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Show the processed frames
        cv2.putText(frame, str(img_text[1]), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0))
        cv2.imshow("test", frame)
        cv2.imshow("mask", mask)

        # Resize the mask to fit model input size and make prediction
        mask_resized = cv2.resize(mask, (IMAGE_X, IMAGE_Y))
        img_text = predictor(mask_resized)
        print(f"Prediction: {img_text[1]}")

        if cv2.waitKey(1) == 27:  # Press 'ESC' to exit
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()