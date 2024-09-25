import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,Dropout
import matplotlib.pyplot as plt
import numpy as np
from keras.api.regularizers import l2
from PIL import Image

train_df = pd.read_csv('sign_mnist_train/sign_mnist_train.csv')
test_df = pd.read_csv('sign_mnist_test/sign_mnist_test.csv')

df=pd.concat(([train_df, test_df]))

label = df['label']
df = df.drop(['label'], axis=1)

X = df.values.astype('float32') / 255
X = X.reshape(-1, 28, 28, 1)

lb = LabelBinarizer()
y = lb.fit_transform(label)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


datagen = ImageDataGenerator(
    rotation_range=10,       # Rotazione casuale fino a 15 gradi
    width_shift_range=0.1,   # Traslazione orizzontale
    height_shift_range=0.1,  # Traslazione verticale
    shear_range=0.0,         # Distorsione (shear)
    zoom_range=0.1,          # Zoom
    horizontal_flip=False,    # Capovolgimento orizzontale
    fill_mode='nearest'      # Riempimento dei pixel vuoti
)

datagen.fit(X_train)

model = Sequential()

num_k = [32, 64]
k_size = [3, 3]
p_size = [2, 2]
num_n = 128
# Aggiungi i layer convoluzionali e di pooling
for i in range(len(num_k)):
    if i == 0:
        model.add(Conv2D(num_k[i], (k_size[i], k_size[i]), activation='relu', input_shape=(28, 28, 1), kernel_regularizer=l2(0.01)))
    else:
        model.add(Conv2D(num_k[i], (k_size[i], k_size[i]), activation='relu', kernel_regularizer=l2(0.01)))
    model.add(MaxPooling2D(pool_size=(p_size[i], p_size[i])))

# Aggiungi i layer densi
model.add(Flatten())
model.add(Dense(num_n, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(24, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

hist = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),  # Usa il generatore di dati augmentati
    steps_per_epoch=len(X_train) // 32,
    epochs=10,
    validation_data=(X_test, y_test)
)

# hist = model.fit(X_train, y_train,
#                     epochs=10,
#                     batch_size=32,
#                     validation_split=0.1)

# Valutazione del modello
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Perdita sul test set: {test_loss:.4f}')
print(f'Accuratezza sul test set: {test_accuracy:.4f}')

# Predizioni
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)


# Visualizzazione di alcune predizioni
num_images = 5
start_index = 5
indices = list(range(start_index, start_index + num_images))
plt.figure(figsize=(15, 3))
for i, idx in enumerate(indices):
    image = X_test[idx].reshape(28, 28)
    true_label = true_classes[idx]
    predicted_label = predicted_classes[idx]
    plt.subplot(1, num_images, i + 1)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title(f'T:{true_label}, P:{predicted_label}')
plt.show()
    
'''
#Grafico Accuratezza
plt.plot(hist.history['accuracy'], label='Accuratezza Training')
plt.plot(hist.history['val_accuracy'], label='Accuratezza Validazione')
plt.xlabel('Epoca')
plt.ylabel('Accuratezza')
plt.legend()
plt.title('Andamento dell\'Accuratezza')
plt.show()

#Grafico Perdita
plt.plot(hist.history['loss'],label='Perdita Training')
plt.plot(hist.history['val_loss'], label='Perdita Validazione')
plt.xlabel('Epoca')
plt.ylabel('Perdita')
plt.legend()
plt.title('Andamento della Perdita')
plt.show()


parola = 'abcdefghiklmnopqrstuvwxy'
img_list = []
for letter in parola:
    img_path = 'test2/' + letter +'.jpg'

    # Apri l'immagine e converti in scala di grigi
    img = Image.open(img_path).convert('L')

    # Ridimensiona a 28x28
    img = img.resize((28, 28))

    # Converti l'immagine in un array numpy
    img_array = np.array(img)

    # Normalizza i pixel tra 0 e 1
    img_array = img_array.astype('float64') / 255.0

    # Aggiungi una dimensione batch (necessario per predict)
    img_array = np.expand_dims(img_array, axis=0)

    # Appiattisci l'array
    img_array = img_array.flatten()
    img_array = img_array.reshape(28, 28, 1)
    img_list.append(img_array)
# Mostra l'immagine

import numpy as np
img_array = np.array(img_list)

predicted_classes = model.predict(img_array)
predicted_classes = np.argmax(predicted_classes, axis = 1)
#print((predicted_classes))

list = 'abcdefghiklmnopqrstuvwxy'
parola_predetta = ''
print(len(list))
for index in range(len(predicted_classes)):

    parola_predetta += list[predicted_classes[index]]

print('Parola da predire:', parola)
print('Parola predetta',parola_predetta)

lettere_riconosciute = []
for index in range(len(parola)):
    if parola[index] == parola_predetta[index]: lettere_riconosciute.append(list[index])
print('Lettere riconosciute = ', len(lettere_riconosciute)/len(parola)*100, '%')
print(lettere_riconosciute)

plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()
'''

# Effettua la predizione
img_path = 'new_test_images/y/hand1_y_bot_seg_4_cropped.jpeg'

# Apri l'immagine e converti in scala di grigi
img = Image.open(img_path).convert('L')

# Ridimensiona a 28x28
img = img.resize((28, 28))

# Converti l'immagine in un array numpy
img_array = np.array(img)

# Normalizza i pixel tra 0 e 1
img_array = img_array.astype('float32') / 255.0

# Aggiungi una dimensione batch (necessario per predict)
img_array = np.expand_dims(img_array, axis=0)

# Aggiungi la dimensione del canale (scala di grigi)
img_array = np.expand_dims(img_array, axis=-1)

# Mostra l'immagine
plt.imshow(img, cmap='gray')  # Utilizza la mappa di colori in scala di grigi
plt.axis('off')  # Nasconde gli assi
plt.show()

prediction2 = model.predict(img_array)

# Estrai l'etichetta predetta
predicted_label2 = np.argmax(prediction2)

print(f'Etichetta predetta: {predicted_label2}')


