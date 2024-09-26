import pandas as pd
from PIL import Image
import numpy as np

df = pd.read_csv('NicolaLavecchia/new_test_images.csv')

print(df.iloc[0].shape)

# Estrai la prima riga
prima_riga = df.iloc[0]

# Estrai i dati dell'immagine (escludi l'etichetta)
img_data = prima_riga[1:].values  # Escludi la prima colonna che contiene l'etichetta

# Ricostruisci l'immagine (28x28)
img_array = img_data.reshape(28, 28)  # Assicurati che l'immagine sia delle dimensioni corrette

# Crea l'immagine utilizzando PIL
img = Image.fromarray(img_array.astype(np.uint8), mode='L')  # 'L' per immagini in scala di grigi

# Visualizza l'immagine
#img.show()
# Mostra il DataFrame


train_df = pd.read_csv('sign_mnist_train/sign_mnist_train.csv')
test_df = pd.read_csv('sign_mnist_test/sign_mnist_test.csv')

df_originale=pd.concat(([train_df, test_df]),axis=0)  

df_originale.to_csv('NicolaLavecchia/old_test_images.csv', index=False)

df_new = pd.concat([df_originale, df],axis=0)

print("Numero righe nuovo: ",df.shape[0])
print("Numero righe originale: ",df_originale.shape[0])
print("Numero righe definitivo: ",df_new.shape[0])

print("Numero colonne nuovo: ",df.iloc[1].shape)
print("Numero colonne originale: ",df_originale.iloc[1].shape)
print("Numero colonne definitivo: ",df_new.iloc[1].shape)
