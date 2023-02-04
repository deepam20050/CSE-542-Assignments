# Deepam Sarmah
# 2020050
# deepam20050@iiitd.ac.in

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

image_size = None
no_of_different_labels = None
image_pixels = None
train_data = None
test_data = None 

def load_mnist ():
  global image_size, no_of_different_labels, image_pixels, train_data, test_data
  image_size = 28 # width and length
  no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
  image_pixels = image_size * image_size
  # train_data = np.loadtxt("./mnist_train.csv", delimiter=",")
  # test_data = np.loadtxt("./mnist_test.csv", delimiter=",") 
  train_data = pd.read_csv("./mnist_train.csv")
  test_data = pd.read_csv('./mnist_train.csv')

def visualize ():
  global train_data
  train_labels = train_data.iloc[:, 0]
  # df.drop(columns=df.columns[0], axis=1,  inplace=True)
  train_data.drop(columns=train_data.columns[0], axis=1, inplace=True)
  # train_data = train_data[:, 1:]
  train_data.drop(index=train_data.index[0], axis=0, inplace=True)

  print(train_data)

  # no_of_rows, no_of_cols = train_data.shape

  # samples = {empty_list : [] for empty_list in range(10)}

  # for i in range(no_of_rows):
  #   label = train_labels[i]
  #   if (len(samples[label]) < 5):
  #     samples[label].append(train_data[i].reshape((28, 28)))

  # plt.figure(figsize = (10, 5))

  # for i in range(10):
  #   for j in range(5):
  #     plt.subplot(10, 5, i * 5 + j + 1)
  #     plt.imshow(samples[i][j], cmap = "Greys")
  #     plt.title('Digit = {}'.format(i))
  # plt.show()

if __name__ == "__main__":
  load_mnist()
  visualize()