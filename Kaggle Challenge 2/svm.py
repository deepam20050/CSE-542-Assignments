import os
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from skimage.io import imread_collection

# Path to training and test image directories
train_dir = 'SML_Train/'
test_dir = 'SML_Test/'

# Load training images
train_images = imread_collection(train_dir + '*.jpg')

# Flatten training images into 1D arrays
train_data = np.array([image.flatten() for image in train_images])

# Load training labels from file
train_labels = np.loadtxt(train_dir + 'labels.txt')

# Create SVM model and fit to training data
svm_model = svm.SVC(kernel='linear', C=1.0)
svm_model.fit(train_data, train_labels)

# Load test images
test_images = imread_collection(test_dir + '*.jpg')

# Flatten test images into 1D arrays
test_data = np.array([image.flatten() for image in test_images])

# Predict labels for test images
test_labels = svm_model.predict(test_data)

# Save test results to CSV file
results_df = pd.DataFrame({'Image': [os.path.basename(image.filename) for image in test_images],
                           'Label': test_labels})
results_df.to_csv('svm_results.csv', index=False)
