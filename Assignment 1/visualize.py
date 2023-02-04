import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset
df = pd.read_csv('mnist_train.csv')

# Create a dictionary to store class labels and their corresponding data
classes = {}
for i in range(10):
    classes[i] = df[df['label'] == i].drop(['label'], axis=1)

# Plot 5 samples from each class
fig, axs = plt.subplots(5, 10, figsize=(15, 6))
axs = axs.ravel()
for i in range(10):
    for j in range(5):
        axs[i * 5 + j].imshow(classes[i].iloc[j].values.reshape(28, 28), cmap='gray')
        axs[i * 5 + j].set_title(f"Class {i}")
        axs[i * 5 + j].axis('off')
plt.show()
