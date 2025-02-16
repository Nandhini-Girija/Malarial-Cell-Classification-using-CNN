import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
import os
import cv2
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize

# Load dataset
image_directory = './dataset/'
SIZE = 150
dataset, label = [], []

# Load parasitized images
parasitized_images = os.listdir(image_directory + 'Parasitized_1/')
for image_name in parasitized_images:
    if image_name.endswith('.png'):
        image = cv2.imread(image_directory + 'Parasitized_1/' + image_name)
        image = Image.fromarray(image, 'RGB').resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append(1)

# Load uninfected images
uninfected_images = os.listdir(image_directory + 'Uninfected_1/')
for image_name in uninfected_images:
    if image_name.endswith('.png'):
        image = cv2.imread(image_directory + 'Uninfected_1/' + image_name)
        image = Image.fromarray(image, 'RGB').resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append(0)

# Convert dataset to NumPy arrays
dataset, label = np.array(dataset), np.array(label)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.20, random_state=0)
X_train, X_test = normalize(X_train, axis=1), normalize(X_test, axis=1)

# Define CNN model
INPUT_SHAPE = (SIZE, SIZE, 3)
model = Sequential([
    Conv2D(32, (3, 3), input_shape=INPUT_SHAPE, activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(32, (3, 3), kernel_initializer='he_uniform', activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), kernel_initializer='he_uniform', activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=64, epochs=100, validation_data=(X_test, y_test), shuffle=False)
model.save('malaria_model.h5')

# Plot loss & accuracy
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss', color='r')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='r')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()
