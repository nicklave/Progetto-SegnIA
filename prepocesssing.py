from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D,MaxPool2D,Dropout
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataProcessor:
    def __init__(self):
        self.train_df=pd.read_csv('sign_mnist_train/sign_mnist_train.csv')
        self.test_df=pd.read_csv('sign_mnist_test/sign_mnist_test.csv')
        

    def train_info(self):
        print('Descrizione dataframe di training')
        print(self.train_df.info())
        print(self.train_df.describe())
        print('Dimensioni:',self.train_df.shape)  

    def test_info(self):
        print('Descrizione dataframe di test')
        print(self.test_df.info())
        print(self.test_df.describe())
        print('Dimensioni:',self.test_df.shape) 

    def train_sample(self):
        print('Esempio di dati nel dataframe di training')
        print(self.train_df.head())

    def test_sample(self):
        print('Esempio di dati nel dataframe di test')
        print(self.test_df.head())

Processor = DataProcessor()
Processor.train_info()
Processor.test_info()
Processor.train_sample()