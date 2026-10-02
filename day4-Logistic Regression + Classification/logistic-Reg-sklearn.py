import numpy as np
from sklearn.linear_model import LogisticRegression


X = np.array([[1] ,[2] ,[3] ,[4] ,[5] ,[6]])
y = np.array([0,0,0,1,1,1])

model = LogisticRegression()

model.fit(X,y)

y_pred = model.predict(X)
y_prob = model.predict_proba(X)

print("Predictions :", y_pred)
print("\nProbablity :", y_prob)
print("\nProbablity Above 0.5 == 1 :", y_prob[:,1])
