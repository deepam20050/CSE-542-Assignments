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
from tensorflow.keras.layers import Dense, MaxPool2D, Conv2D, Dropout, Flatten, GlobalAveragePooling2D, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

# 学習のハイパーパラメータ
EPOCHS = 1000              # 学習回数
hidden_nodes1 = 128        # 中間層ノード数1
hidden_nodes2 = 256        # 中間層ノード数2
hidden_nodes3 = 512        # 中間層ノード数3
output_nodes  = 1024       # 全結合層ノード数
validation_rate = 0.2      # trainデータに対するvalidationデータの割合
IMAGE_SIZE = 64            # 入力画像サイズ
BATCH_SIZE = 500           # 学習する画像枚数

# CNNの構築
model = Sequential()

# 入力層，中間層01
model.add(Conv2D(hidden_nodes1, (3, 3), padding='same', input_shape=(64, 64, 3)))
model.add(Activation('relu'))
model.add(Conv2D(hidden_nodes1, (3, 3), padding='same'))
model.add(Activation('sigmoid'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# 中間層02
model.add(Conv2D(hidden_nodes2, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(hidden_nodes2, (3, 3), padding='same'))
model.add(Activation('sigmoid'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# 中間層03
model.add(Conv2D(hidden_nodes3, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(hidden_nodes3, (3, 3), padding='same'))
model.add(Activation('sigmoid'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# 全結合層
model.add(GlobalAveragePooling2D())
model.add(Dense(output_nodes))
model.add(Activation('relu'))
model.add(Dropout(0.2))

# 10クラスの分類
model.add(Dense(25))
model.add(Activation('softmax'))

# Compile the model
model.compile(optimizer='adam',
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