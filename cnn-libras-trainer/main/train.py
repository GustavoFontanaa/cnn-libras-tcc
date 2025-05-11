from keras.utils import to_categorical, plot_model
from keras.optimizers import SGD
from keras import backend
from sklearn.metrics import classification_report, confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from cnn import Convolution

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import h5py
import time
import os

def getDateStr():
    return str('{date:%d_%m_%Y_%H_%M}').format(date=datetime.datetime.now())

def getTimeMin(start, end):
    return (end - start) / 60

EPOCHS = 50
CLASS = 21
FILE_NAME = 'cnn_model_LIBRAS_'

print("\n\n ----------------------START --------------------------\n")
print('[INFO] [START]: ' + getDateStr())

print('[INFO] Inicializando ImageDataGenerator...')
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.25
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
    shuffle=False,
    class_mode='categorical'
)

print("[INFO] Inicializando e compilando o modelo CNN...")
start = time.time()

early_stopping_monitor = EarlyStopping(
    monitor='val_loss', mode='min', verbose=1, patience=15
)

model = Convolution.build(64, 64, 3, CLASS)
model.compile(optimizer=SGD(0.01), loss="categorical_crossentropy", metrics=["accuracy"])

print("[INFO] Treinando o modelo...")
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

os.makedirs('../models/graphics/', exist_ok=True)
os.makedirs('../models/', exist_ok=True)

file_date = getDateStr()
model_path = '../models/' + FILE_NAME + file_date + '.h5'
model.save(model_path)
print(f'[INFO] Modelo salvo em: {model_path}')

end = time.time()
print("[INFO] Tempo total de execução: %.1f min" % (getTimeMin(start, end)))

print('\n[INFO] Sumário do Modelo:')
model.summary()

print("\n[INFO] Avaliando o modelo no conjunto de teste...")
score = model.evaluate(test_set, steps=(test_set.n // test_set.batch_size), verbose=1)
print('[INFO] Accuracy: %.2f%%' % (score[1] * 100), '| Loss: %.5f' % (score[0]))

print("[INFO] Gerando gráfico de perda e acurácia...")
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), classifier.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), classifier.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), classifier.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), classifier.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('../models/graphics/' + FILE_NAME + file_date + '.png', bbox_inches='tight')

print('[INFO] Gerando matriz de confusão...')
Y_pred = model.predict(test_set, steps=(test_set.n // test_set.batch_size + 1))
y_pred = np.argmax(Y_pred, axis=1)

y_true = test_set.classes

class_indices = training_set.class_indices
index_to_label = {v: k for k, v in class_indices.items()}
labels = [index_to_label[i] for i in range(CLASS)]

cm = confusion_matrix(y_true, y_pred)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

plt.figure(figsize=(12, 10))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão')
plt.xlabel('Classe Predita')
plt.ylabel('Classe Real')
plt.savefig('../models/graphics/' + FILE_NAME + file_date + '_confusion_matrix.png', bbox_inches='tight')
plt.show()

print('\n[INFO] Treinamento Finalizado!!! ')
print('\n\n')
