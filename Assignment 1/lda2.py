import pandas as pd
import numpy as np

# Read MNIST dataset using pandas
df = pd.read_csv("mnist.csv")

# Divide data into features and target
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

# Get unique class labels
class_labels = np.unique(y)

# Compute covariance matrix for each class
cov_matrices = []
for label in class_labels:
    X_class = X[y==label]
    mean_vector = np.mean(X_class, axis=0)
    cov = np.zeros((X_class.shape[1], X_class.shape[1]))
    for i in range(X_class.shape[0]):
        diff = X_class[i]-mean_vector
        cov += np.outer(diff, diff)
    cov /= X_class.shape[0]-1
    cov_matrices.append(cov)

# Compute weighted average covariance matrix
weights = [np.sum(y==label) for label in class_labels]
weights = weights / np.sum(weights)
cov_matrix = np.zeros_like(cov_matrices[0])
for i in range(len(class_labels)):
    cov_matrix += weights[i] * cov_matrices[i]

# Regularize the covariance matrix
cov_matrix += np.eye(X.shape[1]) * 0.01

# Implement LDA function
def linear_discriminant(X_test, class_labels, cov_matrix):
    mean_vectors = []
    for label in class_labels:
        mean_vectors.append(np.mean(X[y==label], axis=0))
    scores = []
    for mean_vector in mean_vectors:
        diff = X_test - mean_vector
        score = np.dot(np.dot(diff, np.linalg.inv(cov_matrix)), diff.T)
        scores.append(score)
    return class_labels[np.argmin(scores)]

# Predict class of samples from testing data
X_test = ... # insert your test data
y_pred = [linear_discriminant(X_test[i], class_labels, cov_matrix) for i in range(X_test.shape[0])]
