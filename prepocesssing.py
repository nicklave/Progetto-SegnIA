from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D,MaxPool2D,Dropout
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer

class DataProcessor:
    def __init__(self):
        self.train_df=pd.read_csv('sign_mnist_train/sign_mnist_train.csv')
        self.test_df=pd.read_csv('sign_mnist_test/sign_mnist_test.csv')
        #divisione etichette e dati
        self.train_label=self.train_df['label']
        trainset=self.train_df.drop(['label'],axis=1)
        self.test_label=self.test_df['label']
        testset=self.test_df.drop(['label'],axis=1)
        #creazione array numpy
        self.X_train = trainset.values
        self.X_test = testset.values
        
        #resize
        self.X_train = self.X_train.reshape(-1,28,28,1)
        self.X_test = self.X_test.reshape(-1,28,28,1)
        #convertire etichette in binario
        lb=LabelBinarizer()
        self.y_train=lb.fit_transform(self.train_label)
        self.y_test=lb.fit_transform(self.test_label)
        

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

    def view_Xtrain(self):
        print(self.X_train.shape)
        print(self.X_train)

    def print_shapes(self):
        print('X_test:', self.X_test.shape)
        print('X_train:', self.X_train.shape)
        print('y_test:', self.y_test.shape)
        print('y_train:', self.y_train.shape)

    def get_datas(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

if __name__ == 'main':
    Processor = DataProcessor()
    Processor.print_shapes()
    # Processor.train_info()
    # Processor.test_info()
    # Processor.train_sample()
    # Processor.view_Xtrain()