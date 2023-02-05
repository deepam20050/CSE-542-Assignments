import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the MNIST dataset from the CSV file
df = pd.read_csv('mnist_train.csv')

# Group by label and sample 5 images per class
grouped = df.groupby('label').apply(lambda x: x.sample(5))

# Display the images and their corresponding label
for i, (index, row) in enumerate(grouped.iterrows()):
    image = row[:-1].values.reshape(28, 28)
    label = row['label']
    plt.subplot(5, 10, i + 1)
    plt.imshow(image, cmap='gray_r')
    plt.axis('off')
    plt.title(label)

plt.tight_layout()
plt.show()

# Compute the covariance matrix for each class
covariance_matrices = []
labels = df['label'].unique()
print(labels)
print(type(labels))
for label in labels:
    class_df = df[df['label'] == label].drop('label', axis=1)
    covariance = np.cov(class_df.T)
    covariance_matrices.append((covariance, class_df.shape[0]))

# Compute the weighted average covariance matrix
weighted_covariance = np.zeros(covariance_matrices[0][0].shape)

total_samples = 0
for covariance, samples in covariance_matrices:
    weighted_covariance += covariance * samples
    total_samples += samples
weighted_covariance /= total_samples

# Assume the weighted average covariance to be the covariance for all classes
covariance = weighted_covariance + np.eye(weighted_covariance.shape[0]) * 1e-5

# Compute the mean for each class
means = {}
for label in labels:
    class_df = df[df['label'] == label].drop('label', axis=1)
    means[label] = np.mean(class_df, axis=0)

# Implement the Linear Discriminant function
def predict(X):
    predictions = []
    for i in range(X.shape[0]):
        max_score = float('-inf')
        max_label = None
        for label, mean in means.items():
            score = np.dot(np.dot((X[i] - mean), np.linalg.inv(covariance)), (X[i] - mean).T)
            if score > max_score:
                max_score = score
                max_label = label
        predictions.append(max_label)
    return predictions

# Use the LDA classifier to make predictions on the test set
X_test = df.drop('label', axis=1).values
y_test = df['label'].values
predictions = predict(X_test)

# Compute the accuracy of the classifier
accuracy = np.mean(predictions == y_test)
print("Accuracy:", accuracy)


def qda(X_train, y_train, X_test):
    classes = np.unique(y_train)
    class_probs = [np.mean(y_train == c) for c in classes]
    class_means = [np.mean(X_train[y_train == c], axis=0) for c in classes]
    class_covariances = [np.cov(X_train[y_train == c].T) for c in classes]
    
    def predict(X):
        predictions = []
        for x in X:
            scores = [np.log(p) - 0.5 * np.log(np.linalg.det(cov)) - 0.5 * np.dot(np.dot((x - mean), np.linalg.inv(cov)), (x - mean).T)
                      for p, mean, cov in zip(class_probs, class_means, class_covariances)]
            predictions.append(classes[np.argmax(scores)])
        return predictions
    
    return predict(X_test)


