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
img.show()
# Mostra il DataFrame
print("Numero righe: ",df.shape[0])