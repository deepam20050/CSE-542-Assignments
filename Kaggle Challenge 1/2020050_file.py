'''
  Deepam Sarmah
  200050
  deepam20050@iiitd.ac.in
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

train_df = pd.read_csv("Train.csv")
test_df = pd.read_csv("Test.csv")

train_features = train_df.drop(["Id", "target"], axis=1)
train_labels = train_df["target"]

trainX, valX, trainY, valY = train_test_split(train_features, train_labels, test_size=0.33, random_state=42)

lgreg = LogisticRegression(random_state=0)
lgreg.fit(trainX, trainY)
predY = lgreg.predict(test_df.drop(["Id"], axis=1))

predictions = pd.DataFrame({'Id': test_df.Id, 'target': predY})
predictions.to_csv("submission.csv", index=False)
joblib.dump(lgreg, "2020050_model.pkl")