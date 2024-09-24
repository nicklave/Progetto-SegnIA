# Classe Modello

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout

class model:

    def __init__(self, num_k, k_size, p_size, num_n, lossfun='categorical_crossentropy'):

        self.model = Sequential()
        self.nclasses = 24

        num_conv_layer = len(num_k)

        # Creazione Layers Convoluzionali
        for i in range(num_conv_layer):
            self.model.add(Conv2D(num_k[i], kernel_size=(k_size[i],k_size[i]), activation='relu', input_shape=(28,28,1)))
            self.model.add(MaxPooling2D(pool_size=(p_size[i],p_size[i])))
        
        # Flattening
        self.model.add(Flatten())

        # Layer Totalmente Connesso
        self.model.add(Dense(num_n, activation='relu'))
        self.model.add(Dropout(rate=0.20))


        # Layer di Output
        self.model.add(Dense(self.nclasses, activation='softmax'))

        # Compilazione del modello
        self.model.compile(optimizer='adam',
                    loss=lossfun,
                    metrics=['accuracy'])