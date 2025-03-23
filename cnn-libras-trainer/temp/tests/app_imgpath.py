import cv2
import numpy as np
import sys
from keras.models import load_model
from keras.preprocessing import image

IMAGE_X, IMAGE_Y = 64, 64
CLASSES = 21
LETTERS = {
    '0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F', '6': 'G',
    '7': 'G', '8': 'I', '9': 'L', '10': 'M', '11': 'N', '12': 'O', '13': 'P',
    '14': 'Q', '15': 'R', '16': 'S', '17': 'T', '18': 'U', '19': 'V', '20': 'W', '21': 'Y'
}

MODEL_PATH = '../../models/cnn_model_LIBRAS_20200531_0304.h5'
classifier = load_model(MODEL_PATH)

def predictor(img):
    test_image = image.img_to_array(img)
    test_image = np.expand_dims(test_image, axis=0)

    # Get the prediction result
    result = classifier.predict(test_image)

    # Find the class with the highest probability
    class_index = np.argmax(result[0])

    # Map the class index to the corresponding letter
    return result, LETTERS[str(class_index)]

def load_and_process_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMAGE_X, IMAGE_Y))
    return img

def display_prediction(image_path, prediction):
    print('\n\n===========================')
    print('Imagem: ', image_path)
    print('Vetor de resultado: ', prediction[0])
    print('Classe: ', prediction[1])
    print('\n===========================')

def main():
    if len(sys.argv) < 2:
        print("Erro: Caminho da imagem não fornecido.")
        sys.exit(1)

    image_path = sys.argv[1]

    # Load and process the image
    img = load_and_process_image(image_path)

    # Make a prediction
    prediction = predictor(img)

    # Display the result
    display_prediction(image_path, prediction)

if __name__ == "__main__":
    main()
