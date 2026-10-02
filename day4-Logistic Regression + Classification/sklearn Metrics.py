from sklearn.metrics import (
    accuracy_score ,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
    )

from sklearn.linear_model import LogisticRegression

import numpy as np

X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8]
])

y = np.array([0,0,0,0,1,1,1,1])

model = LogisticRegression()

model.fit(X,y)
y_predict = model.predict(X)

probabilities = model.predict_proba(X)

print("="*60)
print(f"\nProbabilities : {probabilities}")
print(f"Predictions : {y_predict}")

print(f"\nAccuaracy :", accuracy_score(y , y_predict))
print(f"Precision :", precision_score(y , y_predict))
print(f"Recall :", recall_score(y , y_predict))
print(f"F1 :", f1_score(y , y_predict))
print(f"\nConfusion Matrix\n:", confusion_matrix(y , y_predict))

print(f"\n----Predicting a student studying 4.5 hours will pass or not----\n")
new_student = np.array([[4.5]])
print(model.predict(new_student))
print(model.predict_proba(new_student))