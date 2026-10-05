import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score,
                             confusion_matrix,
                             classification_report
                             )

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

y = np.array([
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1
])

X_train , X_test , y_train , y_test = train_test_split(
    X ,
    y,
    test_size=0.2, 
    random_state=42
)

model = DecisionTreeClassifier( max_depth=3 , random_state=42)

model.fit(X_train , y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test , y_pred)
print("Accuracy:",accuracy)

print("Classification Report :",classification_report(y_test, y_pred))

print("Confusion Matrix :",confusion_matrix(y_test, y_pred))

print("Predictions:", y_pred)
print("Probabilities:")
print(model.predict_proba(X_test))

new_student = np.array([[4.5]])

print("Prediction:", model.predict(new_student))
print("Probability:", model.predict_proba(new_student))


from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))

plot_tree(model,
          feature_names=["Study Hours"],
          class_names=["Fail", "Pass"],
          filled=True
          )

plt.show()
