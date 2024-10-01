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
     
## **Dataset**

Link dataset: https://www.kaggle.com/datasets/datamunge/sign-language-mnist

Il dataset MNIST (Modified National Institute of Standards and Technology) è un ampio e noto database di immagini di cifre scritte a mano, comunemente utilizzato come benchmark nel campo dell'apprendimento automatico e della visione artificiale. Per stimolare ulteriori sviluppi, è stato creato il Sign Language MNIST, che segue lo stesso formato CSV del MNIST, ma rappresenta lettere dell'alfabeto della lingua dei segni americana (ASL) invece delle cifre.

Il dataset contiene 24 classi (lettere dell’alfabeto, escludendo J e Z che richiedono movimenti) ed è organizzato in maniera simile al MNIST, con immagini 28x28 pixel in scala di grigi e valori tra 0-255.

Il train set originale ha 27.455 immagini e quello di test 7.172, ma per garantire una migliore randomizzazione delle classi nel nostro dataset, abbiamo adottato una strategia di concatenazione e successiva suddivisione dei dati, questo contribuisce a una migliore valutazione delle performance del modello, poiché i dati di test riflettono più accuratamente la varietà presente nell'intero dataset.

## **Modello**

- **Tipo di modello**: modello sequenziale (`Sequential`), costruisce la rete neurale layer per layer in modo lineare.

- **Struttura del modello**:

    - *Layer Convoluzionali*: I layer convoluzionali (`Conv2D`) applicano un filtro (o kernel) per estrarre caratteristiche importanti dall’immagine di input. Utilizza la funzione di attivazione *ReLU*, che aiuta a introdurre non linearità nel modello.

    - *Pooling*: il pooling (`MaxPooling2D`) serve per ridurre ulteriormente la dimensione delle caratteristiche e mantenere solo le informazioni più importanti.

    - *Layer Densi*: I dati vengono "appiattiti" (`Flatten`) e passati a uno o più layer densi (`Dense`). Qui, il modello combina le informazioni apprese e le trasforma in output finali.

    - *Dropout*: Il layer di dropout (`Dropout`) serve a prevenire l’overfitting escludendo alcuni neuroni dal processo di addestramento.

- **Compilazione del modello**: Il modello viene compilato specificando l'ottimizzatore (Adam), la funzione di perdita (categorical crossentropy) e la metrica da monitorare (accuratezza). Questo permette al modello di apprendere dai dati in modo efficace.
