import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import os
import matplotlib.pyplot as plt

# Carregar e processar imagens de CAPTCHAs
def load_images(data_dir, img_size=(100, 50)):
    images, labels = [], []
    for filename in os.listdir(data_dir):
        img = cv2.imread(os.path.join(data_dir, filename), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, img_size) / 255.0  # Normalização
        images.append(img.reshape(*img_size, 1))  # Adiciona canal
        labels.append(filename.split("_")[0])  # Supondo que o nome do arquivo seja "1234_xyz.png"
    return np.array(images), np.array(labels)

# Diretório das imagens
data_dir = "dataset/captcha_images/"
images, labels = load_images(data_dir)

# Mapeamento dos caracteres
characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
char_to_index = {char: idx for idx, char in enumerate(characters)}
index_to_char = {idx: char for char, idx in char_to_index.items()}

# Convertendo rótulos para one-hot encoding
labels_encoded = np.array([[char_to_index[c] for c in label] for label in labels])
labels_one_hot = to_categorical(labels_encoded, num_classes=len(characters))

# Criando modelo de CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 50, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(characters), activation='softmax')
])

# Compilar e treinar modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(images, labels_one_hot, epochs=10, batch_size=32, validation_split=0.2)

# Função para prever CAPTCHA
def predict_captcha(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (100, 50)) / 255.0
    img = img.reshape(1, 100, 50, 1)
    prediction = model.predict(img)
    decoded = ''.join(index_to_char[np.argmax(p)] for p in prediction)
    return decoded

# Testando
test_image = "test_captcha.png"
print("Predicted CAPTCHA:", predict_captcha(test_image))
