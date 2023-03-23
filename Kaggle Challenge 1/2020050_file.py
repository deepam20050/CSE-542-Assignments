import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.preprocessing import StandardScaler


train_df = pd.read_csv('Train.csv')
test_df = pd.read_csv('Test.csv')

X_train = train_df.drop('target', axis=1)
y_train = train_df['target']

X_train.drop('Id', inplace=True, axis=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)

# dtc = DecisionTreeClassifier()

# bagging = BaggingClassifier(estimator=dtc, n_estimators=1000, max_samples=150)

# bagging.fit(X_train, y_train)

# gb_clf = GradientBoostingClassifier(loss="exponential", n_estimators=100000, learning_rate=1.95, max_features=10, max_depth=20, random_state=0)
# gb_clf.fit(X_train, y_train)

# nb = GaussianNB()
# nb.fit(X_train, y_train)

# logreg = LogisticRegression(tol=1e-12, solver="liblinear", max_iter=1000000)
# logreg.fit(X_train, y_train)

clf = svm.SVC(kernel='linear', max_iter=-1) # Linear Kernel
clf.fit(X_train, y_train)


y_pred = clf.predict(scaler.transform(test_df.drop("Id", axis=1)))

predictions = pd.DataFrame({'Id': test_df.Id, 'target': y_pred})
predictions.to_csv("submission.csv", index=False)