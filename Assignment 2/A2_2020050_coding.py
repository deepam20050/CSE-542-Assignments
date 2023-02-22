# Deepam Sarmah
# 20200050
# deepam20050@iiitd.ac.in

import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot

train_df, test_df, num_elements = None, None, None

def load_data ():
  global train_df, test_df, num_elements
  train_df = pd.read_csv('mnist_train.csv')
  test_df = pd.read_csv('mnist_test.csv')
  train_df.drop(train_df[train_df['label'] > 1].index, inplace=True)
  test_df.drop(test_df[test_df['label'] > 1].index, inplace=True)
  num_elements = train_df.shape[0]
  np.seterr(divide = 'ignore')
  # Ref: https://stackoverflow.com/a/46093600
  train_x_np = train_df.iloc[:, 1:].to_numpy().astype(np.float32)
  train_y_np = train_df.iloc[:, 0].to_numpy().astype(np.int32)
  test_x_np = test_df.iloc[:, 1:].to_numpy().astype(np.float32)
  test_y_np = test_df.iloc[:, 0].to_numpy().astype(np.int32)
  noise_factor = 0.1
  train_x_noise = train_x_np + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=train_x_np.shape)
  test_x_noise = test_x_np + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=test_x_np.shape)
  train_x_noise = np.clip(train_x_noise, 0.0, 1.0)
  test_x_noise = np.clip(test_x_noise, 0.0, 1.0)
  train_x_noise_df = pd.DataFrame(train_x_noise, columns=train_df.columns[1:])
  test_x_noise_df = pd.DataFrame(test_x_noise, columns=test_df.columns[1:])
  train_y_df = pd.DataFrame(train_y_np, columns=['label'])
  test_y_df = pd.DataFrame(test_y_np, columns=['label'])
  train_df = pd.concat([train_y_df, train_x_noise_df], axis=1)
  test_df = pd.concat([test_y_df, test_x_noise_df], axis=1)
  train_df = train_df.sample(frac=1)
  test_df = test_df.sample(frac=1)

def visualize ():
  samples = train_df.groupby('label').apply(lambda label : label.sample(5))
  for i, (index, row) in enumerate(samples.iterrows()):
    pyplot.subplot(2, 5, i + 1)
    pyplot.imshow(row[: -1].values.reshape(28, 28), cmap = "Greys")
    pyplot.title('Digit = {}'.format(row['label']))
  pyplot.show()

def accuracy_scartch (test_df, label):
  return sum(test_df['label'] == test_df[label]) / test_df.shape[0]

def get_means (train_df):
  label_means = [None] * 2
  for x in range(2):
    label_means[x] = train_df[train_df['label'] == x].mean().drop('label')
  return label_means

# Ref: https://stackoverflow.com/a/8093043
def get_eigen (n, cov):
  eigen_v, eigen_vec = np.linalg.eig(cov)
  eigen_vec = np.vstack((eigen_v, eigen_vec))
  eigen_vec = eigen_vec[eigen_vec[:, 1].argsort()][::-1]
  eigen_vec = np.delete(eigen_vec, 0, axis=0)
  eigen_vec = eigen_vec[:n]
  return eigen_vec

def reconstruct (train_df, train_df_T, e):
  train_df = train_df.T.iloc[1 : , :].T
  train_df_T = train_df_T.T.iloc[1 : , :].T
  rec = (e.T @ train_df_T.T).T
  rec.add(train_df.mean(axis = 0))
  diff = np.array(train_df) - rec
  diff = np.sum(diff ** 2, axis=1)
  diff = np.sum(diff, axis=0) / diff.shape
  return diff[0]

def get_cov (train_df):
  label_covs = [None] * 2
  for x in range(2):
    label_data = train_df[train_df['label'] == x].drop('label', axis=1)
    label_data -= label_data.mean(axis=0)
    label_covs[x] = (label_data.T.dot(label_data)) / (label_data.shape[0] - 1)
  for x in range(2):
    label_covs[x] += np.eye(label_covs[x].shape[0]) * (1e-9)
  weighted_cov = np.zeros_like(label_covs[0])
  for x in range(2):
    weighted_cov += label_covs[x] * train_df[train_df['label'] == x].shape[0]
  weighted_cov /= train_df.shape[0]
  return weighted_cov

def LDA_scratch (train_df, test_df):
  means = get_means(train_df)
  cov = get_cov(train_df)
  cov_inv = np.linalg.inv(cov)
  t1, t2, t3 = [None] * 3, [None] * 3, [None] * 3
  for x in range(2):
    t1[x] = cov_inv @ means[x].T
    t2[x] = 0.5 * means[x].T @ cov_inv @ means[x]
    t3[x] = np.log(train_df[train_df['label'] == x].shape[0] / num_elements)
  def lda (x):
    gi_x = {}
    for label in range(2):
      gi_x[label] = t1[label] @ x - t2[label] + t3[label]
    return max(gi_x, key=gi_x.get)
  test_df['pred_class_lda'] = test_df.drop('label', axis=1).apply(lambda x: lda(x), axis=1)
  print('Scratch LDA Accuracy = ', accuracy_scartch(test_df, 'pred_class_lda'))
  test_df.drop('pred_class_lda', axis=1, inplace=True)

# Using FDA to project data and then using LDA to classify
def fda (train_df, test_df):
  means = get_means(train_df)
  n_cov = [None] * 2
  for i in range(2):
    mat = train_df[train_df["label"] == i].drop("label", axis=1)
    mat -= mat.mean(axis=0)
    n_cov[i] = mat.T.dot(mat)
  sw = n_cov[0] + n_cov[1]
  sw += np.eye(sw.shape[0]) * (1e-9)
  sw_inv = np.linalg.inv(sw)
  w = sw_inv @ (means[0] - means[1])
  train_df_T = w @ train_df.T.iloc[1 : , :]
  train_df_T = pd.concat([train_df.T.iloc[ : 1, :].T, pd.DataFrame(train_df_T[:], columns=['values'])], axis=1)
  test_df_T = w @ test_df.T.iloc[1 : , :]
  test_df_T = pd.concat([test_df.T.iloc[ : 1, :].T, pd.DataFrame(test_df_T[:], columns=['values'])], axis=1)
  LDA_scratch(train_df_T, test_df_T)

# Ref: https://medium.com/@pranjallk1995/pca-for-image-reconstruction-from-scratch-cf4a787c1e36
# Applying pca with best n value from part (e) and then apply FDA, now using LDA to classify testing samples and reporting the accuracy.
def pca_fda_lda (n, train_df, test_df):
  mat = train_df.drop('label', axis=1)
  mat -= mat.mean(axis=0)
  cov = mat.T.dot(mat) / (mat.shape[0] - 1)
  eigen_vec = get_eigen(n, cov)
  train_df_T = eigen_vec @ train_df.T.iloc[1 : , :]
  train_df_T = pd.concat([train_df.T.iloc[ : 1, :], train_df_T[:]]).T
  test_df_T = eigen_vec @ test_df.T.iloc[1 : , :]
  test_df_T = pd.concat([test_df.T.iloc[ : 1, :], test_df_T[:]]).T
  train_df = train_df_T
  test_df = test_df_T
  print("[PCA + FDA + LCA] n = ", n, end=" => ")
  fda(train_df, test_df)

# Ref: https://www.askpython.com/python/examples/principal-component-analysis
# Applying PCA on Input data and using LDA to classify the testing samples.
def pca(n, train_df, test_df):
  mat = train_df.drop('label', axis=1)
  mat -= mat.mean(axis=0)
  cov = mat.T.dot(mat) / (mat.shape[0] - 1)
  eigen_vec = get_eigen(n, cov)
  train_df_T = eigen_vec @ train_df.T.iloc[1 : , :]
  train_df_T = pd.concat([train_df.T.iloc[ : 1, :], train_df_T[:]]).T
  test_df_T = eigen_vec @ test_df.T.iloc[1 : , :]
  test_df_T = pd.concat([test_df.T.iloc[ : 1, :], test_df_T[:]]).T
  print("[PCA] n = ", n, end=" => ")
  LDA_scratch(train_df_T, test_df_T)
  return reconstruct(train_df, train_df_T, eigen_vec)

load_data()
visualize()
n = [2,3,5,8,10,15]
rec_err = [None] * len(n)
for i in range(len(n)):
  rec_err[i] = abs(pca(n[i], train_df, test_df))
pyplot.plot(n, rec_err)
pyplot.show()
print("[FDA] ", end =" => ")
fda(train_df, test_df)
for x in n:
  pca_fda_lda(x, train_df, test_df)