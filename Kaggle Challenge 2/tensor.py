import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

# Define the paths
train_path = 'SML_Train'
test_path = 'SML_Test'
train_csv = 'SML_Train.csv'
test_csv = 'test.csv'

# Read the train CSV file
train_data = pd.read_csv(train_csv)

# Split the dataset into train and validation sets
train_data, val_data = train_test_split(train_data, test_size=0.2)

train_data['category'] = train_data['category'].astype(str)
val_data['category'] = val_data['category'].astype(str)

# Create a data generator for train and validation sets
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale = 1./255,
	rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_data,
    directory=train_path,
    x_col='id',
    y_col='category',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical')

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_data,
    directory=train_path,
    x_col='id',
    y_col='category',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical')

# Build the model
model = tf.keras.models.Sequential([
    # tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
    # tf.keras.layers.MaxPooling2D((2,2)),
    # tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    # tf.keras.layers.MaxPooling2D((2,2)),
    # tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    # tf.keras.layers.MaxPooling2D((2,2)),
    # tf.keras.layers.Flatten(),
    # tf.keras.layers.Dense(512, activation='relu'),
    # tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    # tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    # tf.keras.layers.MaxPooling2D((2, 2)),
    # tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(25, activation='softmax')
])

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator)

model.save('my_model.h5')

# Create a data generator for the test set
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# test_data = pd.DataFrame({'id': os.listdir(test_path), 'category': predicted_classes})

# test_generator = test_datagen.flow_from_directory(
#     # dataframe=test_data,
#     directory=train_path,
#     target_size=(224, 224),
#     batch_size=32,
#     classes=None, 
#     class_mode=None,
#     shuffle=False)

# test_generator = test_datagen.flow_from_directory('.', classes=['SML_Test'])

# # print(train_generator)
# # print(len(test_generator))
# # print(os.listdir(test_path))

# # Make predictions on the test set
# predictions = model.predict(test_generator)

# # Convert the predictions into class labels
# predicted_classes = np.argmax(predictions, axis=1)

# # Create a dataframe with the predicted classes and save it to a CSV file
# test_data = pd.DataFrame({'id': os.listdir(test_path), 'category': predicted_classes})
# test_data.to_csv(test_csv, index=False)

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