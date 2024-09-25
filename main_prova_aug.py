from prepocessing import *
from Modello import *
from Data_augmentation import *
from Conversione_immagini_aug import *
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from keras.utils import to_categorical

Processor = DataProcessor()

X_train,X_test,y_train,y_test = Processor.get_datas()

augmentor = DataAugmentor(X_train,X_test)
datagen = augmentor.augment_data()

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(24, activation='softmax'))  # 24 classi per il dataset Sign MNIST

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

batch_size = 32
epochs = 10

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),  # Usa il generatore di dati augmentati
    steps_per_epoch=len(X_train) // batch_size,
    epochs=epochs,
    validation_data=(X_test, y_test)
)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Perdita sul test set: {test_loss:.4f}')
print(f'Accuratezza sul test set: {test_accuracy:.4f}')

# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label='val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend(loc='lower right')
# plt.show()

img_path = 'test_images/prova2_girata.jpeg'

Converter = ImageConverterAug(img_path)

image_array = Converter.image_array()

# Aggiungi una dimensione batch (necessario per predict)
image_array = np.expand_dims(image_array, axis=0)

# Effettua la predizione
prediction = model.predict(image_array)

# Estrai l'etichetta predetta
predicted_label = np.argmax(prediction)

print(f'Etichetta predetta: {predicted_label}')