import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

IMAGE_X, IMAGE_Y = 64, 64
CLASSES = 21
LETTERS = {
    '0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F', '6': 'G', 
    '7': 'I', '8': 'L', '9': 'M', '10': 'N', '11': 'O', '12': 'P', '13': 'Q', 
    '14': 'R', '15': 'S', '16': 'T', '17': 'U', '18': 'V', '19': 'W', '20': 'Y'
}

MODEL_PATH = '../models/model.h5'
classifier = load_model(MODEL_PATH)

def nothing(x):
    pass

def predictor():
    test_image = load_img('../temp/img.png', target_size=(IMAGE_X, IMAGE_Y))
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    
    # Get the prediction result
    result = classifier.predict(test_image)
    
    # Find the class with the highest probability
    class_index = np.argmax(result[0])

    # Map the class index to the corresponding letter
    return [result, LETTERS[str(class_index)]]

# Initialize webcam
cam = cv2.VideoCapture(0)
img_counter = 0
img_text = ['', '']

while True:
    # Capture frame from webcam
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)  # Flip frame horizontally for mirror effect

    # Define a region of interest (ROI) to crop the hand gesture area
    img = cv2.rectangle(frame, (425, 100), (625, 300), (0, 255, 0), thickness=2)
    imcrop = img[102:298, 427:623]  # Crop the region of interest

    # Display the predicted letter on the main frame
    cv2.putText(frame, str(img_text[1]), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0))
    
    # Show both the original frame and the cropped mask for hand gesture
    cv2.imshow("Test", frame)
    cv2.imshow("Mask", imcrop)

    # Save the cropped image for prediction
    img_name = "../temp/img.png"
    save_img = cv2.resize(imcrop, (IMAGE_X, IMAGE_Y))  # Resize image to model input size
    cv2.imwrite(img_name, save_img)

    # Make prediction and update the displayed text
    img_text = predictor()

    # Display the prediction in a separate window
    output = np.ones((150, 150, 3)) * 255  # Create a white background for output
    cv2.putText(output, str(img_text[1]), (15, 130), cv2.FONT_HERSHEY_TRIPLEX, 6, (255, 0, 0))
    cv2.imshow("Prediction", output)

    if cv2.waitKey(1) == 27:
        break

# Release the webcam and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()