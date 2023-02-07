import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score

train_df, test_df, labels, label_covs, label_means = None, None, None, {}, {}

def load_data ():
  global train_df, test_df, labels
  train_df = pd.read_csv('mnist_train.csv')
  test_df = pd.read_csv('mnist_test.csv')
  labels = np.arange(10)

def visualize ():
  global train_df
  samples = train_df.groupby('label').apply(lambda label : label.sample(5))
  for i, (index, row) in enumerate(samples.iterrows()):
    pyplot.subplot(10, 5, i + 1)
    pyplot.imshow(row[: -1].values.reshape(28, 28), cmap = "Greys")
    pyplot.title('Digit = {}'.format(row['label']))
  pyplot.tight_layout()
  pyplot.show()

def generate_means ():
  global train_df, labels, label_means
  for x in labels:
    label_means[x] = train_df[train_df['label'] == x].mean().drop('label')

def generate_covs ():
  global train_df, labels, label_covs
  for x in labels:
    label_data = train_df[train_df['label'] == x].drop('label', axis=1)
    label_data -= label_data.mean()
    label_covs[x] = (label_data.T.dot(label_data)) / (label_data.shape[0] - 1)
  for x in labels:
    label_covs[x] += np.eye(label_covs[x].shape[0]) * (1e-6)

def generate_weighted_cov ():
  global train_df ,label_covs
  weighted_cov = np.zeros_like(label_covs[0])
  for x in labels:
    weighted_cov += label_covs[x] * train_df[train_df['label'] == x].shape[0]
  weighted_cov /= train_df.shape[0]
  return weighted_cov

def LDA_scratch ():
  global train_df, test_df, labels, label_covs, label_means
  weighted = generate_weighted_cov()
  def lda (x):
    gi_x = {}
    for label in label_means.keys():
      mean = label_means[label].values
      inv = np.linalg.inv(weighted)
      w1 = inv.dot(mean)
      w0 = -0.5 * mean.dot(inv).dot(mean) + np.log(train_df[train_df['label'] == label].shape[0] / train_df.shape[0])
      gi_x[label] = x.dot(w1) + w0
    return max(gi_x, key=gi_x.get)
  test_df['pred_class_lda'] = test_df.drop('label', axis=1).apply(lambda x: lda(x), axis=1)
  lda_accuracy = sum(test_df['label'] == test_df['pred_class_lda']) / test_df.shape[0]
  test_df.drop('pred_class_lda', axis=1, inplace=True)
  print('Scratch LDA Accuracy = ', lda_accuracy)
  np.seterr(divide = 'ignore') 

def QDA_scratch ():
  global train_df, test_df, labels, label_covs, label_means
  def qda(x):
    gi_x = {}
    for label in label_means.keys():
      mean = label_means[label].values
      inv = np.linalg.inv(label_covs[label])
      gi_x[label] = - 0.5 * np.log(np.linalg.det(label_covs[label])) - 0.5 * np.dot(np.dot((x - mean), inv), (x - mean).T) + np.log(train_df[train_df['label'] == label].shape[0] / train_df.shape[0])
    return max(gi_x, key=gi_x.get)
  test_df['pred_class_qda'] = test_df.drop('label', axis=1).apply(lambda x: qda(x), axis=1)
  qda_accuracy = sum(test_df['label'] == test_df['pred_class_qda']) / test_df.shape[0]
  test_df.drop('pred_class_qda', axis=1, inplace=True)
  print('Scratch QDA Accuracy = ', qda_accuracy)

def sk_learn_lda ():
  global train_df, test_df
  lda_df = train_df.sample(frac = 1)
  lda = LinearDiscriminantAnalysis()
  lda.fit(lda_df.drop("label", axis = 1), lda_df["label"])
  lda_results = lda.predict(test_df.drop("label", axis = 1))
  lda_accuracy = accuracy_score(test_df["label"], lda_results)
  print("sk-learn LDA Accuracy = ", lda_accuracy)

def sk_learn_qda ():
  global train_df, test_df
  qda_df = train_df.sample(frac = 1)
  qda = QuadraticDiscriminantAnalysis()
  qda.fit(qda_df.drop("label", axis = 1), qda_df["label"])
  qda_results = qda.predict(test_df.drop("label", axis = 1))
  qda_accuracy = accuracy_score(test_df["label"], qda_results)
  print("sk-learn QDA Accuracy = ", qda_accuracy)

if __name__ == "__main__":
  load_data()
  visualize()
  generate_means()
  generate_covs()
  LDA_scratch()
  QDA_scratch()
  sk_learn_lda()
  sk_learn_qda()