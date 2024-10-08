# Classe Modello

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.api.regularizers import l2


class Model:

    def __init__(self, num_k, k_size, p_size, num_n, lossfun='categorical_crossentropy'):

        self.model = Sequential()
        self.nclasses = 24

        num_conv_layer = len(num_k)

        # Creazione Layers Convoluzionali
        for i in range(num_conv_layer):
            if i == 0:
                self.model.add(Conv2D(num_k[i], (k_size[i], k_size[i]), activation='relu', input_shape=(28, 28, 1), kernel_regularizer=l2(0.01)))
            else:
                self.model.add(Conv2D(num_k[i], (k_size[i], k_size[i]), activation='relu', kernel_regularizer=l2(0.01)))
            self.model.add(MaxPooling2D(pool_size=(p_size[i], p_size[i])))
        # Flattening
        self.model.add(Flatten())

        # Layer Totalmente Connesso
        self.model.add(Dense(num_n, activation='relu'))
        #self.model.add(Dropout(0.5))
        # Layer di Output
        self.model.add(Dense(self.nclasses, activation='softmax'))

        # Compilazione del modello
        self.model.compile(optimizer='adam',
                    loss=lossfun,
                    metrics=['accuracy'],
                    )#shuffle=1