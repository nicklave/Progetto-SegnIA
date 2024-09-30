from prepocessing import DataProcessor
from Data_augmentation import DataAugmentor
from Modello import Model
from Training import Training
from Testing import Testing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.api.regularizers import l2
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
Processor = DataProcessor()

X_train, X_test, X_val, y_val, y_train, y_test = Processor.get_datas()

Augmentor = DataAugmentor(X_train, X_test)

datagen = Augmentor.augment_data()

num_k = [32, 64]
k_size = [3, 3]
p_size = [2, 2]
num_n = 128

MyModel = Model(num_k, k_size, p_size, num_n)

Trainer = Training(MyModel.model, datagen, X_train, y_train, ep = 10, val_data=(X_val,y_val))
trained_model = Trainer.get_trained_model()

Tester = Testing(X_test, y_test, trained_model)

predicted_classes, true_classes = Testing.predictions(X_test, y_test)

Testing.confusionmatrix(true_classes, predicted_classes)

