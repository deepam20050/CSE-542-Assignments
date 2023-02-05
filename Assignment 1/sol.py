import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score

train_df, test_df = None, None

def load_data ():
  global train_df, test_df
  train_df = pd.read_csv('mnist_train.csv')
  test_df = pd.read_csv('mnist_test.csv')

def visualize ():
  global train_df
  samples = train_df.groupby('label').apply(lambda label : label.sample(5))
  for i, (index, row) in enumerate(samples.iterrows()):
    pyplot.subplot(10, 5, i + 1)
    pyplot.imshow(row[: -1].values.reshape(28, 28), cmap = "Greys")
    pyplot.title('Digit = {}'.format(row['label']))
  pyplot.tight_layout()
  pyplot.show()

def accuracy (A, B):
  

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
  sk_learn_lda()
  sk_learn_qda()