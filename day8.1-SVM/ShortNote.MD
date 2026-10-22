# Day 8 — Support Vector Machine (SVM)

## 1. What is SVM?

Support Vector Machine (SVM) is a supervised machine learning algorithm used mainly for classification. It can also be used for regression.

**Common uses:**

- Spam email classification
- Image classification
- Text classification
- Classifying data with complex decision boundaries

## 2. Important concepts

- **Decision boundary:** The line or hyperplane that separates classes.
- **Margin:** The gap between the decision boundary and the closest training points.
- **Support vectors:** Training points closest to the boundary that help determine its position.
- **Kernel:** Determines the type of decision boundary the SVM can learn.
- **C:** Controls how strongly margin violations are penalized.

## 3. SVC — Basic implementation

`SVC` is scikit-learn's Support Vector Classification implementation.

```python
from sklearn.svm import SVC

model = SVC(kernel="linear", C=1)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

- `fit()` trains the model.
- `predict()` predicts labels for new inputs.

## 4. SVM kernels

| Kernel    | Purpose                                  |
| --------- | ---------------------------------------- |
| `linear`  | Learns a linear decision boundary        |
| `rbf`     | Can learn nonlinear boundaries           |
| `poly`    | Models polynomial relationships          |
| `sigmoid` | Provides another nonlinear kernel option |

Example:

```python
SVC(kernel="linear", C=1)
SVC(kernel="rbf", C=1)
SVC(kernel="poly", degree=3)
SVC(kernel="sigmoid")
```

Start by comparing `linear` and `rbf`.

## 5. Understanding C

`C` controls the penalty for margin violations.

- **Low C:** More tolerant of violations; may produce a wider margin.
- **High C:** Penalizes violations more strongly; may fit training data more closely.
- **C = 1:** A convenient starting point, not necessarily the best value.

Example:

```python
for c in [0.01, 0.1, 1, 10, 100]:
    model = SVC(kernel="linear", C=c)
    model.fit(X_train, y_train)
    print(c, model.score(X_test, y_test))
```

Use validation or cross-validation to select hyperparameters rather than relying only on training accuracy.

## 6. Feature scaling

SVM can be sensitive to feature scales. `StandardScaler` standardizes features using the training data's mean and standard deviation.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Important: use `fit_transform()` on training data and only `transform()` on test data to avoid data leakage.

## 7. Using make_pipeline()

A pipeline combines preprocessing and model training.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

model = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", C=1)
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

The pipeline automatically scales the data before training and prediction.

## 8. Train/test split and evaluation

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Actual:", y_test)
print("Predictions:", predictions)
print("Accuracy:", accuracy_score(y_test, predictions))
```

### Remember the workflow

1. `X_train`, `y_train` → training data and labels.
2. `model.fit(X_train, y_train)` → train.
3. `model.predict(X_test)` → predict.
4. `accuracy_score(y_test, predictions)` → evaluate.

## 9. My SVM practice result

For the final practice dataset:

 #### <a>day8.2-train_and_test/practice.ipynb</a>

```text
Actual:      [1 1 0]
Predictions: [1 1 0]
Accuracy:    1.0
```

All three test predictions were correct. Since the test set contained only three samples, this result does not guarantee the same performance on real-world data.
