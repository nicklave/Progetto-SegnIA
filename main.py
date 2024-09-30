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
from Conversione_immagini import ImageConverter

#-----------Pre-Processing dei dati--------------------------
Processor = DataProcessor()
X_train, X_test, X_val, y_val, y_train, y_test = Processor.get_datas()
#-----------Data Augmentation-----------------------------------
Augmentor = DataAugmentor(X_train, X_test)
datagen = Augmentor.augment_data()
#----------Parametri modello------------------------------------
num_k = [32, 64]
k_size = [3, 3]
p_size = [2, 2]
num_n = 128
#---------Costruzione del modello-------------------------------
MyModel = Model(num_k, k_size, p_size, num_n)
#--------Training del modello-----------------------------------
Trainer = Training(MyModel.model, datagen, X_train, y_train, ep = 10, val_data=(X_val,y_val))
trained_model = Trainer.get_trained_model()
Trainer.grafici_accuracy_loss()

#----------Testing del modello----------------------------------
Tester = Testing(X_test, y_test, trained_model)
Tester.print_metrics()
predicted_classes = Tester.predictions(X_test= X_test)
true_classes = np.argmax(y_test, axis=1)
Tester.confusionmatrix(true_classes, predicted_classes)

#---------Tentativo riconoscimento immagini nuove---------------------------------
#---------------------------------------------------------------------------------

#---------Primo Dataset-----------------------------------------------------------
img_path = 'test_images/'
lista = 'abcdefghiklmnopqrstuvwxy'
parola = 'abcdefghiklmnopqrstuvwxy'
img_list = []
for letter in parola:
    Converter = ImageConverter(img_path+letter.upper()+'_test.jpg')
    #Converter.show_image()
    img_list.append(Converter.image_array().reshape(28,28,1))
    

img_array = np.array(img_list)

predicted_classes = Tester.predictions(img_array)

parola_predetta = ''

for index in range(len(predicted_classes)):

    parola_predetta += lista[predicted_classes[index]]

print('Parola da predire:', parola)
print('Parola predetta',parola_predetta)

lettere_riconosciute1 = []
for index in range(len(parola)):
    if parola[index] == parola_predetta[index]: lettere_riconosciute1.append(lista[index])
print('Lettere riconosciute = ', len(lettere_riconosciute1)/len(parola)*100, '%')
print(lettere_riconosciute1)
#----------------------------------------------------------------------------
#------------------Secondo Dataset-------------------------------------------
#----------------------------------------------------------------------------
img_path = 'test2/'
lista = 'abcdefghiklmnopqrstuvwxy'
parola = 'abcdefghiklmnopqrstuvwxy'
img_list = []
for letter in parola:
    Converter = ImageConverter(img_path+letter+'.jpeg')
    #Converter.show_image()
    img_list.append(Converter.image_array().reshape(28,28,1))
    

img_array = np.array(img_list)

predicted_classes = Tester.predictions(img_array)

parola_predetta = ''

for index in range(len(predicted_classes)):

    parola_predetta += lista[predicted_classes[index]]

print('Parola da predire:', parola)
print('Parola predetta',parola_predetta)

lettere_riconosciute = []
for index in range(len(parola)):
    if parola[index] == parola_predetta[index]: lettere_riconosciute.append(lista[index])
print('Lettere riconosciute = ', len(lettere_riconosciute)/len(parola)*100, '%')
print(lettere_riconosciute)