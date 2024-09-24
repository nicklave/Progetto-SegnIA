from PIL import Image
import matplotlib.pyplot as plt

img_path = r'C:\\Users\\nicol\\Desktop\\Data science Experis\\Project work\\A_test.jpg'

img = Image.open(img_path).convert('L')  # Converti in scala di grigi

img = img.resize((28, 28))  # Ridimensiona a 28x28

import numpy as np
img_array = np.array(img)  # Converti l'immagine in un array numpy
img_array = img_array / 255.0  # Normalizza i pixel tra 0 e 1

img_array = img_array.flatten()  # Appiattisci in un array 1D di 784 elementi

plt.imshow(img, cmap='gray')  # Utilizza la mappa di colori in scala di grigi
plt.axis('off')  # Nasconde gli assi
plt.show()  # Mostra l'immagine