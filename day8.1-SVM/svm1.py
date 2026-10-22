from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import accuracy_score

X = ([
  [1], [2], [3], [4],[5], [6], [7], [8]
])

y = ([0, 0, 0, 0, 1, 1, 1, 1])

model = SVC(kernel="linear",C=1)

model.fit(X , y)

print(model.predict([[7]]))

print("\nSupport Vectors :")
print(model.support_vectors_)

print("\nSupport Vectors indices :")
print(model.support_)

print()
# for c in [0.01, 0.1, 1, 10, 100]:
#     model = SVC(kernel="linear", C=c)
#     model.fit(X, y)

#     predictions = model.predict(X)

#     print(
#         "C:", c,
#         "| Accuracy:", accuracy_score(y, predictions)
#     )

from sklearn.model_selection import train_test_split

X_train , X_test , y_train , y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

model = SVC(kernel="linear" , C=1)
model.fit(X,y)

predictions = model.predict(X_test)

print("Actual:", y_test)
print("predction:",predictions)
print("Test accuracy:",accuracy_score(y_test , predictions))