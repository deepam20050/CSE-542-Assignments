import pandas as pd
import cupy as cp

# Load the MNIST dataset in csv format
train_df = pd.read_csv("mnist_train.csv")
test_df = pd.read_csv("mnist_test.csv")

# Convert the pandas dataframes to cupy arrays
train_data = cp.array(train_df.values)
test_data = cp.array(test_df.values)

# Extract the features and labels
train_features = train_data[:, :-1]
train_labels = train_data[:, -1].astype(int)

test_features = test_data[:, :-1]
test_labels = test_data[:, -1].astype(int)

# Number of classes
num_classes = len(cp.unique(train_labels))

# Compute the mean of each class
class_means = []
for c in range(num_classes):
    class_data = train_features[train_labels == c, :]
    class_mean = cp.mean(class_data, axis=0)
    class_means.append(class_mean)

# Compute the covariance of each class
class_covs = []
for c in range(num_classes):
    class_data = train_features[train_labels == c, :]
    class_mean = class_means[c]
    cov = cp.zeros((class_data.shape[1], class_data.shape[1]), dtype=cp.float32)
    for i in range(class_data.shape[0]):
        x = class_data[i, :] - class_mean
        cov += cp.dot(x.reshape(-1, 1), x.reshape(1, -1))
    cov /= class_data.shape[0]
    class_covs.append(cov)

# Compute the weighted average covariance matrix
class_weights = []
for c in range(num_classes):
    class_data = train_features[train_labels == c, :]
    class_weights.append(class_data.shape[0] / train_features.shape[0])

sigma_g = cp.zeros(class_covs[0].shape, dtype=cp.float32)
for c in range(num_classes):
    sigma_g += class_weights[c] * class_covs[c]

# Handle singular covariance matrix by adding a regularization term
reg_term = 1e-3 * cp.eye(sigma_g.shape[0], dtype=cp.float32)
sigma_g += reg_term

# Implement the discriminant function
discriminant_scores = []
for x in test_features:
    scores = []
    for c in range(num_classes):
        mean = class_means[c]
        inv_sigma_g = cp.linalg.inv(sigma_g)
        w = cp.dot(inv_sigma_g, mean)
        b = -0.5 * cp.dot(cp.dot(mean.T, inv_sigma_g), mean) + cp.log(class_weights[c])
        score = cp.dot(w, x) + b
        scores.append(score)
    discriminant_scores.append(scores)