# Classe Training

import matplotlib.pyplot as plt

class training:

    def __init__(self, model, X_train, y_train, ep, b_size, val_split):

        self.trainset=X_train
        self.target=y_train

        self.trained_model=model

        self.hist=self.trained_model.fit(self.trainset, self.target,
                    epochs=ep,
                    batch_size=b_size,
                    validation_split=val_split)

    def grafico_accuracy(self):
        plt.plot(self.hist.history['accuracy'],
        label='Accuratezza Training')
        plt.plot(self.hist.history['val_accuracy'],
        label='Accuratezza Validazione')
        plt.xlabel('Epoca')
        plt.ylabel('Accuratezza')
        plt.legend()
        plt.title('Andamento dell\'Accuratezza')
        plt.show()
        
    def grafico_loss(self):
        plt.plot(self.hist.history['loss'],
        label='Perdita Training')
        plt.plot(self.hist.history['val_loss'], label='Perdita Validazione')
        plt.xlabel('Epoca')
        plt.ylabel('Perdita')
        plt.legend()
        plt.title('Andamento della Perdita')
        plt.show()
