'''
  Deepam Sarmah
  2020050
  deepam20050@iiitd.ac.in
'''

import pandas as pd
import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models

# Define the paths
train_path = 'SML_Train'
test_path = 'SML_Test'
train_csv = 'SML_Train.csv'
test_csv = 'test.csv'

# Read the train CSV file
train_data = pd.read_csv(train_csv)

# Split the dataset into train and validation sets

train_data['category'] = train_data['category'].astype(str)

# Create a data generator for train and validation sets
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    # rescale = 1./255,
	# rotation_range=40,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # shear_range=0.2,
    # zoom_range=0.2,
    # horizontal_flip=True,
    featurewise_center=True, featurewise_std_normalization=True)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_data,
    directory=train_path,
    x_col='id',
    y_col='category',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical')

# Create the CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (2, 2), activation='relu', input_shape=(64, 64, 3)))
model.add(layers.AveragePooling2D((2, 2)))
model.add(layers.Conv2D(64, (2, 2), activation='sigmoid'))
model.add(layers.AveragePooling2D((2, 2)))
model.add(layers.Conv2D(64, (2, 2), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='sigmoid'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(25, activation='softmax'))

# Compile the model
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=10)

# Create the testing data
test_files = os.listdir('SML_Test')
X_test = []
for file in test_files:
    img_path = os.path.join('SML_Test', file)
    img = cv2.imread(img_path)
    # img = cv2.resize(img, img_size)
    X_test.append(img)

# Convert the testing data to numpy arrays
X_test = np.array(X_test)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Save the predictions to a CSV file
test_df = pd.DataFrame({'id': test_files, 'category': np.argmax(y_pred, axis=1)})
test_df.to_csv('SML_Test_Predictions.csv', index=False)