# Classe Testing

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


class Testing:

    def __init__(self, X_test, y_test, model):

        self.tested_model = model
        self.test_loss, self.test_accuracy = self.tested_model.evaluate(X_test, y_test)

    def print_metrics(self):
        print('Test Loss:', self.test_loss)
        print('Test Accuracy', self.test_accuracy)
        

    def predictions(self, X_test):

        predictions = self.tested_model.predict(X_test)

        # Converte le probabilità nelle classi predette dal modello
        predicted_classes = np.argmax(predictions, axis=1)
        # Estrae le etichette di classe corrette dal test set
        #true_classes = np.argmax(y_test, axis=1)

        return predicted_classes#, true_classes


    def confusionmatrix(self, true_classes, predicted_classes):
        
        # Matrice di confusione
        conf_matrix = confusion_matrix(true_classes, predicted_classes)

        # Visualizzazione della matrice di confusione
        plt.figure(figsize=(10,8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Matrice di Confusione')
        plt.xlabel('Predizione')
        plt.ylabel('Vero Valore')
        plt.show()
