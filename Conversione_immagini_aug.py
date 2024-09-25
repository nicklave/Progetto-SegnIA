from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

class ImageConverterAug:
    def __init__(self, path):
        img_path = path

        self.img = Image.open(img_path).convert('L')  # Converti in scala di grigi

        self.img = self.img.resize((28, 28))  # Ridimensiona a 28x28


        self.img_array = np.array(self.img)  # Converti l'immagine in un array numpy
        self.img_array = self.img_array.astype('float64') / 255.0  # Normalizza i pixel tra 0 e 1

        self.img_array = np.expand_dims(self.img_array, axis=-1)

    def image_array(self):
        return self.img_array

    def show_image(self):
        plt.imshow(self.img, cmap='gray')  # Utilizza la mappa di colori in scala di grigi
        plt.axis('off')  # Nasconde gli assi
        plt.show()
