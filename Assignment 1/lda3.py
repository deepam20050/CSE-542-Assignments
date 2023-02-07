import pandas as pd
import numpy as np

# read MNIST dataset
df = pd.read_csv('mnist_train.csv')

df = df.head(50)

# split into training and testing data
df_train = df.sample(frac=0.8, random_state=0)
df_test = df.drop(df_train.index)

# compute mean for each label
classes = df_train['label'].unique()
class_means = {}
for c in classes:
    class_means[c] = df_train[df_train['label'] == c].mean().drop('label')

# compute covariance for each label
class_covs = {}
for c in classes:
    class_data = df_train[df_train['label'] == c].drop('label', axis=1)
    class_data = class_data - class_data.mean()
    class_covs[c] = (class_data.T.dot(class_data)) / (class_data.shape[0]-1)

# add a small constant to the diagonal of each covariance matrix to make it non-singular
eps = 1e-5
for c in classes:
    class_covs[c] += np.eye(class_covs[c].shape[0]) * eps

# compute weighted average of covariance for all classes
weighted_avg_cov = np.zeros_like(class_covs[0])
for c in classes:
    weighted_avg_cov += class_covs[c] * df_train[df_train['label'] == c].shape[0]
weighted_avg_cov /= df_train.shape[0]

# implement the Linear Discriminant function
def lda(x, class_means, weighted_avg_cov):
    scores = {}
    for c in class_means.keys():
        mean = class_means[c].values
        inv_cov = np.linalg.inv(weighted_avg_cov)
        w = inv_cov.dot(mean)
        b = -0.5 * mean.dot(inv_cov).dot(mean) + np.log(df_train[df_train['label'] == c].shape[0] / df_train.shape[0])
        scores[c] = x.dot(w) + b
    return max(scores, key=scores.get)

# predict label of samples from testing data
df_test['pred_class'] = df_test.drop('label', axis=1).apply(lambda x: lda(x, class_means, weighted_avg_cov), axis=1)

# calculate accuracy of predictions
accuracy = sum(df_test['label'] == df_test['pred_class']) / df_test.shape[0]
print('Accuracy:', accuracy)
