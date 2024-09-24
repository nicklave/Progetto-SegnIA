import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np

train_df = pd.read_csv('sign_mnist_train/sign_mnist_train.csv')
test_df = pd.read_csv('sign_mnist_test/sign_mnist_test.csv')

df=pd.concat(([train_df, test_df]))

label = df['label']
df = df.drop(['label'], axis=1)

X = df.values.astype('float') / 255
X = X.reshape(-1, 28, 28, 1)

lb = LabelBinarizer()
y = lb.fit_transform(label)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train_label = train_df['label']
# trainset = train_df.drop(['label'], axis=1)
# test_label = test_df['label']
# testset = test_df.drop(['label'], axis=1)

# X_train = trainset.values.astype('float') / 255
# X_test = testset.values.astype('float') / 255

# X_train = X_train.reshape(-1, 28, 28, 1)
# X_test = X_test.reshape(-1, 28, 28, 1)

# lb = LabelBinarizer()
# y_train = lb.fit_transform(train_label)
# y_test = lb.transform(test_label)

model = Sequential()

num_k = [32, 64]
k_size = [3, 3]
p_size = [2, 2]
num_n = 128
# Aggiungi i layer convoluzionali e di pooling
for i in range(len(num_k)):
    if i == 0:
        model.add(Conv2D(num_k[i], (k_size[i], k_size[i]), activation='relu', input_shape=(28, 28, 1)))
    else:
        model.add(Conv2D(num_k[i], (k_size[i], k_size[i]), activation='relu'))
    model.add(MaxPooling2D(pool_size=(p_size[i], p_size[i])))

# Aggiungi i layer densi
model.add(Flatten())
model.add(Dense(num_n, activation='relu'))
model.add(Dense(24, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

hist = model.fit(X_train, y_train,
                    epochs=5,
                    batch_size=32,
                    validation_split=0.1)

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
indices = list(range(num_images))
plt.figure(figsize=(15, 3))
for i in indices:
    image = X_test[i].reshape(28, 28)
    true_label = true_classes[i]
    predicted_label = predicted_classes[i]
    plt.subplot(1, num_images, i + 1)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title(f'T:{true_label}, P:{predicted_label}')
plt.show()
    

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