from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LeakyReLU

class Convolution:
    @staticmethod
    def build(width, height, channels, classes):
        """
        Builds a CNN model with the following architecture:
        INPUT => CONV => POOL => CONV => POOL => CONV => POOL => FC => FC => OUTPUT
        
        :param width: Image width in pixels.
        :param height: Image height in pixels.
        :param channels: Number of image channels.
        :param classes: Number of output classes.
        :return: Compiled CNN model.
        """
        input_shape = (height, width, channels)
        model = Sequential()

        # First convolutional layer
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D((2, 2)))

        # Second convolutional layer
        model.add(Conv2D(32, (3, 3)))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D((2, 2)))

        # Third convolutional layer
        model.add(Conv2D(64, (3, 3)))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D((2, 2)))

        # Fully connected layers
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(classes, activation='softmax'))

        return model
