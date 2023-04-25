import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model

# Define the ResNet block
def resnet_block(inputs, filters, kernel_size, strides):
    x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    if strides != 1:
        shortcut = Conv2D(filters, kernel_size=1, strides=strides, padding='same')(inputs)
        shortcut = BatchNormalization()(shortcut)
    else:
        shortcut = inputs
    x = Add()([x, shortcut])
    x = ReLU()(x)
    return x

# Define the ResNet model
def resnet(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, kernel_size=7, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = AveragePooling2D(pool_size=3, strides=2, padding='same')(x)
    x = resnet_block(x, filters=64, kernel_size=3, strides=1)
    x = resnet_block(x, filters=64, kernel_size=3, strides=1)
    x = resnet_block(x, filters=128, kernel_size=3, strides=2)
    x = resnet_block(x, filters=128, kernel_size=3, strides=1)
    x = resnet_block(x, filters=256, kernel_size=3, strides=2)
    x = resnet_block(x, filters=256, kernel_size=3, strides=1)
    x = resnet_block(x, filters=512, kernel_size=3, strides=2)
    x = resnet_block(x, filters=512, kernel_size=3, strides=1)
    x = resnet_block(x, filters=512, kernel_size=1, strides=1) # add another ResNet block with a stride of 2 and a kernel size of 1 before the AveragePooling2D layer
    x = AveragePooling2D(pool_size=4)(x)
    x = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Define the model parameters and create the ResNet model
input_shape = (64, 64, 3)
num_classes = 25
model = resnet(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Create the data generators for the training and testing images
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    'SML_Train',
    target_size=(input_shape[0], input_shape[1]),
    batch_size=32,
    class_mode='categorical',
    subset='training')

validation_generator = train_datagen.flow_from_directory(
    'SML_Train',
    target_size=(input_shape[0], input_shape[1]),
    batch_size=32,
    class_mode='categorical',
    subset='validation')

# Train the model
model.fit(train_generator,
          epochs=10,
          validation_data=validation_generator)

# Create the testing data
test_files = os.listdir('SML_Test')
X_test = []
for file in test_files:
    img_path = os.path.join('SML_Test', file)
    img = cv2.imread(img_path)
    # img = cv2.resize(img, 224)
    X_test.append(img)

# Convert the testing data to numpy arrays
X_test = np.array(X_test)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Save the predictions to a CSV file
test_df = pd.DataFrame({'id': test_files, 'category': np.argmax(y_pred, axis=1)})
test_df.to_csv('SML_Test_Predictions.csv', index=False)