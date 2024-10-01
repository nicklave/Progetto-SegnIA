# Progetto-SegnIA
Project Work per Crif a cura di Aldo Perri, Martina Curcio, Maurizio Mattei, Nicola Lavecchia.
Per una prova veloce usare main_notebook.ipynb.

Il progetto si propone di sviluppare un sistema di riconoscimento del linguaggio dei segni utilizzando tecniche di computer vision e intelligenza artificiale. 

L'obiettivo principale è creare un'applicazione capace di interpretare e tradurre automaticamente i segni eseguiti con le mani in testi, facilitando così la comunicazione tra le persone sorde e chi non conosce il linguaggio dei segni. 

Il sistema sfrutta reti neurali convoluzionali (CNN) per l'analisi delle immagini, identificando i gesti e associandoli alle corrispondenti parole o frasi. Questo approccio può essere integrato in dispositivi mobili o applicazioni web, rendendo il riconoscimento accessibile in vari contesti della vita quotidiana.

## **Librerie**

- `pandas` : manipolazione e analisi del dataset

- `matplotlib`: 
    - `pyplot`: visualizzare dati e risultati dei modelli

- `numpy`: gestione di vettori, matrici e array multidimensionali

- `seaborn`: creazione di grafici statistici

- `scikit-learn` : 
    - `preprocessing` - `LabelBinarizer` : conversione delle etichette in array binario
    - `model_selection` - `train_test_split` : divisione del dataset in train e test set
    - `metrics` - `confusion_matrix` :  matrice di confusione

- `keras` : 
    - `models`-`Sequential`: costruzione di una rete neurale in modo lineare
    - `layers`:
        - `Conv2D`: convoluzione bidimensionale
        - `MaxPooling2D`: pooling
        - `Flatten`: appiattimento dell'input in un vettore unidimensionale
        - `Dense`: strato completamente connesso
        - `Dropout`:  disattivazione casuale una frazione di neuroni durante l'addestramento
