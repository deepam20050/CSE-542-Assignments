import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the MNIST dataset
df = pd.read_csv("mnist.csv")

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df.drop("label", axis=1), df["label"], test_size=0.2)

# Calculate the mean of each class
means = []
classes = np.unique(y_train)
for c in classes:
    X_class = X_train[y_train == c]
    means.append(np.mean(X_class, axis=0))
means = np.array(means)

# Calculate the covariance matrix for all classes
covariance = np.zeros((X_train.shape[1], X_train.shape[1]))
for c in classes:
    X_class = X_train[y_train == c]
    for x in X_class:
        covariance += np.outer((x - means[c]), (x - means[c]).T)
covariance /= X_train.shape[0]

# Make predictions on the test data
predictions = []
for x in X_test:
    scores = []
    for c in range(len(classes)):
        mean = means[c]
        score = np.dot(np.dot((x - mean), np.linalg.inv(covariance)), (x - mean).T)
        scores.append(-0.5 * score)
    predictions.append(classes[np.argmax(scores)])

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, predictions)
print("LDA accuracy:", accuracy)
