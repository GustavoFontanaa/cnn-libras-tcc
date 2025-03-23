from keras.utils import to_categorical, plot_model
from keras.optimizers import SGD, Adam
from keras import backend
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from cnn import Convolution

import datetime
import h5py
import time

def getDateStr():
    return str('{date:%d_%m_%Y_%H_%M}').format(date=datetime.datetime.now())

def getTimeMin(start, end):
    return (end - start) / 60

EPOCHS = 20
CLASS = 21
FILE_NAME = 'cnn_model_LIBRAS_'

print("\n\n ----------------------START --------------------------\n")
print('[INFO] [START]: ' + getDateStr())
print('[INFO] Downloading dataset using keras.preprocessing.image.ImageDataGenerator')

train_datagen = ImageDataGenerator(
    rescale=1./255,        # Rescale pixel values to [0,1]
    shear_range=0.2,       # Shear intensity (rotation in degrees)
    zoom_range=0.2,        # Random zoom range
    horizontal_flip=True,  # Randomly flip images horizontally
    validation_split=0.25  # Split validation set (25%)
)

test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.05)

training_set = train_datagen.flow_from_directory(
    '../dataset/training',
    target_size=(64, 64),
    color_mode='rgb',
    batch_size=32,
    shuffle=True,
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    '../dataset/test',
    target_size=(64, 64),
    color_mode='rgb',
    batch_size=32,
    shuffle=True,
    class_mode='categorical'
)

print("[INFO] Initializing and optimizing the CNN model...")
start = time.time()

# Early stopping callback to stop training if there's no improvement in validation loss
early_stopping_monitor = EarlyStopping(
    monitor='val_loss', mode='min', verbose=1, patience=15
)

model = Convolution.build(64, 64, 3, CLASS)

model.compile(optimizer=SGD(0.01), loss="categorical_crossentropy", metrics=["accuracy"])

print("[INFO] Training the CNN model...")
classifier = model.fit(
    training_set,
    steps_per_epoch=(training_set.n // training_set.batch_size),
    epochs=EPOCHS,
    validation_data=test_set,
    validation_steps=(test_set.n // test_set.batch_size),
    verbose=2,
    callbacks=[early_stopping_monitor]
)

EPOCHS = len(classifier.history["loss"])

# Save the trained model
print("[INFO] Saving trained model...")
file_date = getDateStr()
model.save('../models/' + FILE_NAME + file_date + '.h5')
print('[INFO] Model saved at: ../models/' + FILE_NAME + file_date + '.h5')

end = time.time()
print("[INFO] CNN execution time: %.1f min" % (getTimeMin(start, end)))

print('[INFO] Model Summary:')
model.summary()

print("\n[INFO] Evaluating the CNN model...")
score = model.evaluate_generator(generator=test_set, steps=(test_set.n // test_set.batch_size), verbose=1)
print('[INFO] Accuracy: %.2f%%' % (score[1] * 100), '| Loss: %.5f' % (score[0]))

print("[INFO] Plotting loss and accuracy graphs...")
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), classifier.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), classifier.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), classifier.history["acc"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), classifier.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('../models/graphics/' + FILE_NAME + file_date + '.png', bbox_inches='tight')

print('[INFO] Generating CNN layer architecture image...')
plot_model(model, to_file='../models/image/' + FILE_NAME + file_date + '.png', show_shapes=True)

print('\n[INFO] [END]: ' + getDateStr())
print('\n\n')
