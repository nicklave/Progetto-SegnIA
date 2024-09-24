from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D,MaxPool2D,Dropout
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataAugmentor:
    def __init__(self, X_train,X_test):
       

        self.X_train = X_train.reshape(-1, 28, 28) 
        self.X_train = np.expand_dims(self.X_train, axis=-1) 

        self.X_test = X_test.reshape(-1, 28, 28)
        self.X_test = np.expand_dims(self.X_test, axis=-1)


        self.datagen = ImageDataGenerator(
            rotation_range=15,       # Rotazione casuale fino a 15 gradi
            width_shift_range=0.1,   # Traslazione orizzontale
            height_shift_range=0.1,  # Traslazione verticale
            shear_range=0.1,         # Distorsione (shear)
            zoom_range=0.1,          # Zoom
            horizontal_flip=True,    # Capovolgimento orizzontale
            fill_mode='nearest'      # Riempimento dei pixel vuoti
        )
        self.X_train = X_train

    def augment_data(self):
        # Fit the data augmentation generator (richiesto da Keras)
        self.datagen.fit(self.X_train)
        return self.datagen