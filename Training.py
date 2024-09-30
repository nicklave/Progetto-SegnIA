# Classe Training

import matplotlib.pyplot as plt

class Training:

    def __init__(self, model, datagen,  X_train, y_train, ep,  val_data, b_size = 32):
        steps_per_epoch = len(X_train) // b_size
        self.trainset=X_train.reshape(-1,28,28,1)
        self.target=y_train
        
        self.model=model
        train_generator = datagen.flow(self.trainset, y_train, batch_size=b_size)
        self.hist=self.model.fit(train_generator,
                                 steps_per_epoch=steps_per_epoch,
                                    epochs=ep,
                                    validation_data = val_data)

    def get_trained_model(self):
        return self.model

    def grafici_accuracy_loss(self):
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # 1 riga, 2 colonne, dimensione della figura 12x5
        
        # Grafico dell'accuratezza
        axs[0].plot(self.hist.history['accuracy'], label='Accuratezza Training')
        axs[0].plot(self.hist.history['val_accuracy'], label='Accuratezza Validazione')
        axs[0].set_xlabel('Epoca')
        axs[0].set_ylabel('Accuratezza')
        axs[0].legend()
        axs[0].set_title('Andamento dell\'Accuratezza')

        # Grafico della perdita
        axs[1].plot(self.hist.history['loss'], label='Perdita Training')
        axs[1].plot(self.hist.history['val_loss'], label='Perdita Validazione')
        axs[1].set_xlabel('Epoca')
        axs[1].set_ylabel('Perdita')
        axs[1].legend()
        axs[1].set_title('Andamento della Perdita')
    
        plt.tight_layout()  # Per gestire bene gli spazi tra i grafici
        plt.show()
