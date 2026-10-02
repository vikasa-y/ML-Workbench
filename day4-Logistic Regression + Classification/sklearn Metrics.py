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

print("="*120)
print(f"\nProbabilities : {probabilities}")
print(f"Predictions : {y_predict}")

print("\nAccuaracy :", accuracy_score(y , y_predict))
print("Precision :", precision_score(y , y_predict))
print("Recall :", recall_score(y , y_predict))
print("F1 :", f1_score(y , y_predict))
print("\nConfusion Matrix\n:", confusion_matrix(y , y_predict))

print("="*120)
print(f"----Predicting Weather student will pass or Fail----\n")

new_student = np.array([
    [4.5],
    [6]
    ])

predictions = model.predict(new_student)
probabilities = model.predict_proba(new_student)[:,1]

for hours,prediction,probability in zip(
    new_student.flatten(), 
    predictions, 
    probabilities
    ):
    print(
        f"{hours} hours :"
        f"{'Pass' if prediction == 1 else 'Fail'}"
        f"({probability:.2%} Pass Probability)"
    )
