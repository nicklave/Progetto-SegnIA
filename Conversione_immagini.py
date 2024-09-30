from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

class ImageConverter:
    def __init__(self, path):
        img_path = path

        self.img = Image.open(img_path) .convert('L')  # Converti in scala di grigi

        self.img = self.img.resize((28, 28))  # Ridimensiona a 28x28


        self.img_array = np.array(self.img)  # Converti l'immagine in un array numpy
        self.img_array = self.img_array.astype('float64') / 255.0  # Normalizza i pixel tra 0 e 1

        self.img_array = self.img_array.flatten()

    def image_array(self):
        return self.img_array

    def show_image(self):
        plt.imshow(self.img, cmap='gray')  # Utilizza la mappa di colori in scala di grigi
        plt.axis('off')  # Nasconde gli assi
        plt.show()

if __name__ == '__main__':
    img_path = 'test_images/prova3.jpeg'

    path = 'test_images/F_test.jpg'
    Converter = ImageConverter(path)
    Converter.show_image()
    img_array = Converter.image_array()
    img_array = img_array.reshape(1,28,28,1)
    img_array.shape