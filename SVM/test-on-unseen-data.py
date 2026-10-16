from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , classification_report
import numpy as np

# Center points: class 0
# Outer points: class 1
X = np.array([
    [0, 0], [0.2, 0.1], [-0.2, 0.1],
    [0.1, -0.2], [-0.1, -0.2],
    [2, 2], [2, -2], [-2, 2], [-2, -2],
    [2, 0], [-2, 0], [0, 2], [0, -2]
])

y = np.array([
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1
])


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

model = SVC(kernel="rbf", C=1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Predictions:", y_pred)
print("Actual:", y_test)
print("Test accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))
