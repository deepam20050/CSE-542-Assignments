import pandas as pd
import cupy as cp
import numpy as np

def LDA(X_train, y_train, X_test):
    # Extract number of classes and number of features
    num_classes = len(np.unique(y_train))
    num_features = X_train.shape[1]

    # Calculate the mean vector for each class
    mean_vectors = []
    for class_ in range(num_classes):
        mean_vectors.append(cp.mean(X_train[y_train==class_], axis=0))

    # Calculate the covariance matrix for each class
    class_cov_matrices = []
    for class_ in range(num_classes):
        class_mean = cp.mean(X_train[y_train==class_], axis=0)
        class_cov_matrices.append(cp.dot(cp.transpose(X_train[y_train==class_] - class_mean), 
                                         X_train[y_train==class_] - class_mean) / len(y_train[y_train==class_]))

    # Calculate the weighted average of covariance matrices
    weight = cp.zeros((num_classes, 1))
    for class_ in range(num_classes):
        weight[class_] = cp.sum(y_train==class_) / len(y_train)
    total_cov_matrix = cp.zeros((num_features, num_features))
    for class_ in range(num_classes):
        total_cov_matrix += weight[class_] * class_cov_matrices[class_]

    # Regularize the covariance matrix to ensure its invertibility
    total_cov_matrix = total_cov_matrix + cp.eye(num_features) * 10e-6

    # Predict the class for each sample in the test set
    y_pred = cp.zeros((X_test.shape[0], 1))
    for i in range(X_test.shape[0]):
        discriminant_values = []
        for class_ in range(num_classes):
            mean = mean_vectors[class_]
            inv = cp.linalg.inv(total_cov_matrix)
            w1 = inv.dot(mean)
            w0 = -0.5 * mean.dot(inv).dot(mean) + np.log(y_train[y_train['label' == class_]] / y_train.size)
            discriminant_values.append(X_test[i, :].dot(w1) + w0)
        y_pred[i] = cp.argmax(cp.asarray(discriminant_values))

    return y_pred

# Load the training and testing data
train_data = pd.read_csv("mnist_train.csv")
test_data = pd.read_csv("mnist_test.csv")

# Split the training data into feature and label arrays
X_train = cp.array(train_data.drop("label", axis=1))
y_train = cp.array(train_data["label"])

# Split the testing data into feature and label arrays
X_test = cp.array(test_data.drop("label", axis=1))
y_test = cp.array(test_data["label"])

y_pred = LDA(X_train, y_train, X_test)
print(cp.count_nonzero(y_test == y_pred) / len(y_test))
