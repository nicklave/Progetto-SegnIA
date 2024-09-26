import numpy as np
from PIL import Image
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

'''
parola = 'abcdefghiklmnopqrstuvwxy'
img_list = []
label_list = []

# Definisci l'assegnazione delle etichette saltando il 9
labels = list(range(9)) + list(range(10, 25))

for idx, letter in enumerate(parola):
    # Definisci il percorso dell'immagine per la lettera corrente
    img_path = f'new_test_images/{letter}/hand1_{letter}_bot_seg_4_cropped.jpeg'

    # Apri l'immagine e converti in scala di grigi
    img = Image.open(img_path).convert('L')

    # Ridimensiona l'immagine a 28x28
    img = img.resize((28, 28))

    # Converti l'immagine in un array numpy
    img_array = np.array(img)

    # Appiattisci l'immagine in un vettore di 784 elementi
    img_array = img_array.flatten()

    # Aggiungi l'immagine elaborata alla lista
    img_list.append(img_array)

    # Aggiungi l'etichetta corrispondente alla lettera
    label_list.append(labels[idx])

# Crea un DataFrame da img_list e aggiungi la colonna delle etichette
df = pd.DataFrame(img_list)
df.insert(0, 'label', label_list)

#df.to_csv('NicolaLavecchia/new_test_images.csv', index=False, header=False)
'''

train_df = pd.read_csv('sign_mnist_train/sign_mnist_train.csv')
test_df = pd.read_csv('sign_mnist_test/sign_mnist_test.csv')

df=pd.concat(([train_df, test_df]),axis=0)

# Estrai le immagini dal DataFrame nuovo
img_arrays = df.iloc[:, 1:].values  # Escludi la colonna delle etichette

img_arrays = img_arrays.reshape(-1, 28, 28, 1)  # Aggiungi dimensione canale per il formato (batch, altezza, larghezza, canali)

# Definisci il generatore di immagini
datagen = ImageDataGenerator(
    rotation_range=10,       # Rotazione casuale fino a 10 gradi
    width_shift_range=0.1,   # Traslazione orizzontale
    height_shift_range=0.1,  # Traslazione verticale
    shear_range=0.0,         # Distorsione (shear)
    zoom_range=0.1,          # Zoom
    horizontal_flip=False,    # Capovolgimento orizzontale
    fill_mode='nearest'      # Riempimento dei pixel vuoti
)

# Configura il numero di augmentazioni necessarie
target_rows = 10000
current_rows = len(df)
augmentations_needed = (target_rows // current_rows) + 1  # +1 per garantire di avere almeno 5000 righe

augmented_images = []

# Genera immagini augmentate
for _ in range(augmentations_needed):
    for img in img_arrays:
        img = img.reshape((1,) + img.shape)  # Aggiungi dimensione batch
        # Genera 100 varianti per ogni immagine
        for augmented in datagen.flow(img, batch_size=1):
            augmented_images.append(augmented[0].astype(np.uint8))  # Augmented[0] contiene l'immagine augmentata
            if len(augmented_images) >= 100:  # Limita a 100 varianti per immagine
                break  # Esci dal ciclo se hai generato abbastanza immagini

# Crea un nuovo DataFrame con le immagini augmentate
augmented_df = pd.DataFrame(np.array(augmented_images).reshape(-1, 784))  # Riscrivi le immagini in formato 784

augmented_df.insert(0, 'label', np.tile(df['label'].values, augmentations_needed * 100)[:len(augmented_df)])  # Aggiungi le etichette

# Crea i nomi delle colonne da 'pixel1' a 'pixel784'
pixel_columns = [f'pixel{i}' for i in range(1, 785)]

# Rinominare le colonne del DataFrame (mantieni la prima colonna come 'Label')
augmented_df.columns = ['label'] + pixel_columns


# Salva il DataFrame in un file CSV
augmented_df.to_csv('NicolaLavecchia/new_test_images.csv', index=False)



train_df = pd.read_csv('sign_mnist_train/sign_mnist_train.csv')
print("Original ",train_df.shape[0])