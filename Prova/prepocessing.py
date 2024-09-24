
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelBinarizer



class DataProcessor:
    def __init__(self):

         #caricamento dati di test e training
        self.train_df=pd.read_csv('sign_mnist_train/sign_mnist_train.csv')
        self.test_df=pd.read_csv('sign_mnist_test/sign_mnist_test.csv')

        #divisione etichette e dati
        self.df=pd.concat(([self.train_df, self.test_df]))

        self.target=self.df['label']
        self.df=self.df.drop(['label'],axis=1)

        #creazione array numpy normalizzati
        self.df = self.df.values.astype('float') / 255

        #resize
        self.df = self.df.reshape(-1,28,28,1)


        #convertire etichette in binario
        lb=LabelBinarizer()
        self.target=lb.fit_transform(self.target)

    def get_datas(self):
        return self.df, self.target
