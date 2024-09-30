from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D,MaxPool2D,Dropout
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split



class DataProcessor:
    def __init__(self, color = False):
        #caricamento dati di test e training       
        self.train_df=pd.read_csv('sign_mnist_train/sign_mnist_train.csv')
        self.test_df=pd.read_csv('sign_mnist_test/sign_mnist_test.csv')
        self.train_label = self.train_df['label']
        df=pd.concat(([self.train_df, self.test_df]))

        label = df['label']
        df = df.drop(['label'], axis=1)

        X = df.values.astype('float') / 255
        self.X = X.reshape(-1, 28, 28, 1)

        lb = LabelBinarizer()
        self.y = lb.fit_transform(label)

        X_train, self.X_test, y_train, self.y_test = train_test_split(X, self.y, test_size=0.2)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        self.X_train = self.X_train.reshape(-1,28,28,1)
        self.X_test = self.X_test.reshape(-1,28,28,1)
        self.X_val = self.X_val.reshape(-1,28,28,1)



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
        #print(set(self.train_df['label'].values))

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

    def view_images(self):
        letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i','j','k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y']
        for index in range(5):
            plt.imshow(self.X_train[index], cmap = 'gray')
            plt.colorbar()
            plt.title(f"Immagine generata dalla matrice di grigi label: {self.train_label[index]} lettera: {letters[self.train_label[index]]}")
            plt.show()

    def frequency_plot(self):
        sns.countplot(x = self.train_label, palette= 'Set2', stat='count')
        plt.title("Frequency of each label")
        plt.show()

    def get_datas(self):

        return self.X_train, self.X_test, self.X_val, self.y_val, self.y_train, self.y_test


if __name__ == '__main__':
    Processor = DataProcessor()
    Processor.print_shapes()
    Processor.view_images()
    Processor.train_sample()
    Processor.frequency_plot()

    # Processor.test_info()
    # Processor.train_sample()
    #Processor.view_Xtrain()