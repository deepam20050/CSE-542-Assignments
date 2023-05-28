'''
  Deepam Sarmah
  2020050
  deepam20050@iiitd.ac.in
  Best
'''

import numpy as np
import os
import tensorflow as tf
import cv2
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the CSV file
train_df = pd.read_csv('SML_Train.csv')

# Define the image size and number of classes
img_size = (64, 64)
num_classes = 25

mean = 0
var = 50
sigma = var ** 0.5
# Create the training data
X_train = []
y_train = []
for idx, row in train_df.iterrows():
    img_path = os.path.join('SML_Train', str(row['id']))
    img = cv2.imread(img_path)
    noise = np.random.normal(mean, sigma, img.shape)
    noised = img + noise
    X_train.append(img)
    X_train.append(noised)
    y_train.append(row['category'])
    y_train.append(row['category'])

# Convert the training data to numpy arrays
X_train = np.array(X_train)
y_train = tf.keras.utils.to_categorical(y_train, num_classes)

# Define VGG16 model
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(64, 64, 3)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dense(4096, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile model with specified optimizer, loss function, and metric
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10)

# Create the testing data
test_files = os.listdir('SML_Test')
X_test = []
for file in test_files:
    img_path = os.path.join('SML_Test', file)
    img = cv2.imread(img_path)
    img = cv2.resize(img, img_size)
    X_test.append(img)

# Convert the testing data to numpy arrays
X_test = np.array(X_test)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Save the predictions to a CSV file
test_df = pd.DataFrame({'id': test_files, 'category': np.argmax(y_pred, axis=1)})
test_df.to_csv('SML_Test_Predictions.csv', index=False)

model.save("2020050_model")