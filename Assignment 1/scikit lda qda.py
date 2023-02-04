import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the MNIST dataset
df = pd.read_csv("mnist_train.csv")

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df.drop("label", axis=1), df["label"], test_size=0.2)

# Fit a LDA model to the training data
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# Predict the classes of the test data
lda_predictions = lda.predict(X_test)

# Calculate the accuracy of the LDA model
lda_accuracy = accuracy_score(y_test, lda_predictions)
print("LDA accuracy:", lda_accuracy)

# Fit a QDA model to the training data
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)

# Predict the classes of the test data
qda_predictions = qda.predict(X_test)

# Calculate the accuracy of the QDA model
qda_accuracy = accuracy_score(y_test, qda_predictions)
print("QDA accuracy:", qda_accuracy)
