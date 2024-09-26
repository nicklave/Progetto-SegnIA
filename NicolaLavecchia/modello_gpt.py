import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Carica i dati
train_df = pd.read_csv('sign_mnist_train/sign_mnist_train.csv')
test_df = pd.read_csv('sign_mnist_test/sign_mnist_test.csv')

df = pd.concat([train_df, test_df])

label = df['label']
df = df.drop(['label'], axis=1)

X = df.values.astype('float32') / 255
X = X.reshape(-1, 28, 28, 1)

lb = LabelBinarizer()
y = lb.fit_transform(label)

# Divisione train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest'
)
datagen.fit(X_train)

# Definizione del modello con miglioramenti
model = Sequential()

# Aggiungi più blocchi di convoluzioni
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), kernel_regularizer=l2(0.001)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

# Layer denso e dropout per prevenire overfitting
model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.5))

model.add(Dense(24, activation='softmax'))

# Compilazione del modello
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callback per interrompere l'allenamento se non migliora e ridurre il learning rate
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Training con data augmentation e callback
hist = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    steps_per_epoch=len(X_train) // 32,
    epochs=10,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, reduce_lr]
)

# Valutazione del modello
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Perdita sul test set: {test_loss:.4f}')
print(f'Accuratezza sul test set: {test_accuracy:.4f}')

# Caricamento delle immagini e predizioni
parola = 'abcdefghiklmnopqrstuvwxy'
img_list = []

for letter in parola:
    img_path = f'new_test_images/{letter}/hand1_{letter}_bot_seg_4_cropped.jpeg'
    try:
        img = Image.open(img_path).convert('L')
        img = img.resize((28, 28))
        img_array = np.array(img).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.expand_dims(img_array, axis=-1)
        img_list.append(img_array)
    except FileNotFoundError:
        print(f"File non trovato per la lettera: {letter}")
        img_list.append(None)  # Aggiungi un placeholder per le immagini mancanti

# Previsioni solo per le immagini valide
img_list = [img for img in img_list if img is not None]  # Filtra le immagini valide
if len(img_list) == 0:
    print("Nessuna immagine trovata.")
else:
    img_array = np.vstack(img_list)  # Combina le immagini in un array
    predicted_classes = model.predict(img_array)
    predicted_classes = np.argmax(predicted_classes, axis=1)

    # Costruzione della parola predetta
    list = 'abcdefghiklmnopqrstuvwxy'
    parola_predetta = ''.join([list[pred] for pred in predicted_classes])

    # Stampa i risultati
    print('Parola da predire:', parola)
    print('Parola predetta:', parola_predetta)

    lettere_riconosciute = [parola[i] for i in range(len(parola)) if parola[i] == parola_predetta[i]]
    print('Lettere riconosciute =', len(lettere_riconosciute) / len(parola) * 100, '%')
    print('Lettere riconosciute:', lettere_riconosciute)

    # Visualizza l'ultima immagine
    if img is not None:
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.show()
