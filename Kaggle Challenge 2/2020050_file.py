'''
  Deepam Sarmah
  2020050
  deepam20050@iiitd.ac.in
  Best
'''

import pandas as pd
import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models

# Load the CSV file
train_df = pd.read_csv('SML_Train.csv')

# Define the image size and number of classes
img_size = (64, 64)
num_classes = 25

# Create the training data
X_train = []
y_train = []
for idx, row in train_df.iterrows():
    img_path = os.path.join('SML_Train', str(row['id']))
    img = cv2.imread(img_path)
    X_train.append(img)
    y_train.append(row['category'])

# Convert the training data to numpy arrays
X_train = np.array(X_train)
y_train = tf.keras.utils.to_categorical(y_train, num_classes)

# Create the CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (2, 2), activation='relu', input_shape=(img_size[0], img_size[1], 3)))
model.add(layers.AveragePooling2D((2, 2)))
model.add(layers.Conv2D(64, (2, 2), activation='sigmoid'))
model.add(layers.AveragePooling2D((2, 2)))
model.add(layers.Conv2D(64, (2, 2), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='sigmoid'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='rmsprop',
              loss="categorical_crossentropy",
              metrics=['accuracy'])

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