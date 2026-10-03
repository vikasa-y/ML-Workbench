# ML Day 4 — Logistic Regression & Classification

## 📌 Goal

Learn how to solve **classification problems** using Logistic Regression.

---

# 1. What is Classification?

Classification means predicting a **category/class** instead of a continuous numerical value.

### Examples

| Problem           | Output              |
| ----------------- | ------------------- |
| Student result    | Pass / Fail         |
| Email             | Spam / Not Spam     |
| Transaction       | Fraud / Normal      |
| Customer          | Churn / Stay        |
| Disease detection | Positive / Negative |

### Binary Classification

When there are only two classes:

```text
0 → Fail
1 → Pass
```

---

# 2. Why Not Linear Regression?

Linear Regression can produce any numerical value:

```text
-5
0.7
2
15
100
```

But classification needs something that can represent probability:

```text
0 → 0%
1 → 100%
```

Therefore, Logistic Regression uses the **Sigmoid function**.

---

# 3. Logistic Regression

The basic equation is:

```text
z = wx + b
```

Then we pass `z` through the sigmoid function.

```text
p = sigmoid(z)
```

Complete flow:

```text
X
↓
z = wx + b
↓
Sigmoid
↓
Probability
↓
Threshold
↓
Class 0 / 1
```

---

# 4. Sigmoid Function

Formula:

```text
σ(z) = 1 / (1 + e⁻ᶻ)
```

The sigmoid converts any real number into a value between:

```text
0 and 1
```

### Examples

```text
z = -5 → probability ≈ 0.007
z =  0 → probability = 0.5
z =  5 → probability ≈ 0.993
```

So:

```text
Large negative z → probability close to 0
z = 0             → probability 0.5
Large positive z → probability close to 1
```

---

# 5. Classification Threshold

Usually we use:

```text
Probability >= 0.5 → Class 1
Probability <  0.5 → Class 0
```

Example:

```text
Probability = 0.82
↓
0.82 >= 0.5
↓
Class 1
```

The threshold can be changed depending on the problem.

---

# 6. Logistic Regression Mathematics

For one feature:

```text
z = wx + b
```

For multiple features:

```text
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

Then:

```text
p = 1 / (1 + e⁻ᶻ)
```

Example:

```text
x = 4
w = 1
b = -3

z = (1)(4) - 3
z = 1
```

Then:

```text
sigmoid(1) ≈ 0.731
```

So the model predicts approximately:

```text
73.1% probability of class 1
```

Since:

```text
0.731 > 0.5
```

prediction:

```text
Class 1
```

---

# 7. Log Loss / Binary Cross-Entropy

Logistic Regression predicts probabilities, so we need a loss function suitable for probabilities.

For one sample:

```text
L = -[y log(p) + (1-y) log(1-p)]
```

For the complete dataset:

```text
J = -(1/n) Σ [yᵢ log(pᵢ) + (1-yᵢ) log(1-pᵢ)]
```

### Intuition

If the prediction is confidently correct:

```text
Actual = 1
Prediction = 0.99
↓
Small loss
```

If the prediction is confidently wrong:

```text
Actual = 1
Prediction = 0.01
↓
Very large loss
```

Therefore:

> **Log Loss strongly penalizes confident wrong predictions.**

---

# 8. Gradient Descent for Logistic Regression

The training process is:

```text
X
↓
z = wx + b
↓
Sigmoid
↓
Probability
↓
Log Loss
↓
Calculate gradient
↓
Update w and b
↓
Repeat
```

Gradient formulas:

```text
dw = (1/n) Σ xᵢ(pᵢ - yᵢ)

db = (1/n) Σ(pᵢ - yᵢ)
```

Parameter updates:

```text
w = w - αdw

b = b - αdb
```

Where:

```text
α = learning rate
```

---

# 9. Logistic Regression From Scratch

Basic sigmoid:

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

Training idea:

```python
n = len(X)

w = 0
b = 0

learning_rate = 0.1

for epoch in range(1000):

    z = w * X + b

    p = sigmoid(z)

    dw = (1/n) * np.sum(X * (p - y))
    db = (1/n) * np.sum(p - y)

    w = w - learning_rate * dw
    b = b - learning_rate * db
```

Prediction:

```python
z = w * X + b

probability = sigmoid(z)

y_pred = (probability >= 0.5).astype(int)
```

---

# 10. Logistic Regression with Scikit-Learn

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X, y)
```

Make predictions:

```python
y_pred = model.predict(X)
```

Get probabilities:

```python
probabilities = model.predict_proba(X)
```

---

# 11. `predict()` vs `predict_proba()`

### `predict()`

Returns the final class.

```python
model.predict(X)
```

Example:

```text
[0 0 1 1]
```

### `predict_proba()`

Returns the probability of each class.

```python
model.predict_proba(X)
```

Example:

```text
[[0.90, 0.10],
 [0.70, 0.30],
 [0.20, 0.80],
 [0.05, 0.95]]
```

Each row:

```text
[Probability of class 0, Probability of class 1]
```

To get only class-1 probability:

```python
model.predict_proba(X)[:, 1]
```

---

# 12. Decision Boundary

The standard classification threshold is:

```text
p = 0.5
```

For sigmoid:

```text
p = 0.5
```

corresponds to:

```text
z = 0
```

Therefore:

```text
wx + b = 0
```

For one feature:

```text
x = -b / w
```

Example:

```text
w = 2
b = -6

2x - 6 = 0

x = 3
```

So approximately:

```text
x < 3 → Class 0
x > 3 → Class 1
```

With multiple features:

```text
w₁x₁ + w₂x₂ + b = 0
```

This forms a decision boundary.

---

# 13. Classification Metrics

## Confusion Matrix

A confusion matrix contains:

```text
                 Predicted
                 0       1

Actual 0        TN      FP

Actual 1        FN      TP
```

### True Positive — TP

Actual = 1
Predicted = 1

### True Negative — TN

Actual = 0
Predicted = 0

### False Positive — FP

Actual = 0
Predicted = 1

### False Negative — FN

Actual = 1
Predicted = 0

---

# 14. Accuracy

Measures the percentage of all predictions that are correct.

```text
Accuracy =
(TP + TN) /
(TP + TN + FP + FN)
```

Use:

```python
accuracy_score(y, y_pred)
```

---

# 15. Precision

Precision asks:

> Of everything the model predicted as positive, how many were actually positive?

```text
Precision = TP / (TP + FP)
```

Use:

```python
precision_score(y, y_pred)
```

Important when **False Positives are costly**.

---

# 16. Recall

Recall asks:

> Of all the actual positive cases, how many did the model find?

```text
Recall = TP / (TP + FN)
```

Use:

```python
recall_score(y, y_pred)
```

Important when **False Negatives are costly**.

---

# 17. F1 Score

F1 combines Precision and Recall.

```text
      2 × (Precision × Recall)
 F1 = -------------------------
      Precision + Recall
```

Use:

```python
f1_score(y, y_pred)
```

Useful when we want a balance between precision and recall.

---

# 18. Accuracy Can Be Misleading

Suppose a fraud dataset has:

```text
1000 transactions

980 → Normal
20  → Fraud
```

A model predicts everything as normal.

Accuracy:

```text
980 / 1000 = 98%
```

Looks excellent.

But:

```text
Fraud detected = 0
```

Therefore, for imbalanced classification problems, we should look beyond accuracy.

---

# 19. Mini Project — Student Pass/Fail Prediction

Dataset:

```python
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

y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
```

Here:

```text
X → Study hours
y → Pass/Fail

0 → Fail
1 → Pass
```

Train:

```python
model = LogisticRegression()

model.fit(X, y)
```

Predict:

```python
y_predict = model.predict(X)
```

Probabilities:

```python
probabilities = model.predict_proba(X)
```

Metrics:

```python
accuracy_score(y, y_predict)

precision_score(y, y_predict)

recall_score(y, y_predict)

f1_score(y, y_predict)

confusion_matrix(y, y_predict)
```

Predict new students:

```python
new_student = np.array([
    [4.5],
    [6]
])

predictions = model.predict(new_student)

probabilities = model.predict_proba(new_student)[:, 1]
```

For multiple students:

```python
for hours, prediction, probability in zip(
    new_student.flatten(),
    predictions,
    probabilities
):

    if prediction == 1:
        result = "Pass"
    else:
        result = "Fail"

    print(f"{hours} hours: {result}")
    print(f"Pass Probability: {probability:.2%}")
```

---

# 20. `flatten()`
flatten converts 2D array into 1D array , its numpy function
<br>
If:

```python
new_student = np.array([
    [4.5],
    [6]
])
```

Shape:

```text
(2, 1)
```

`flatten()` converts it to:

```text
[4.5 6.]
```

Shape:

```text
(2,)
```

Remember:

> **`flatten()` converts an array into 1D.**

---

# 21. Day 4 Mental Model

```text
                 CLASSIFICATION
                       │
                       ▼
                       X
                       │
                       ▼
                  wx + b
                       │
                       ▼
                   Sigmoid
                       │
                       ▼
                  Probability
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
        predict_proba()       Threshold
                                  │
                                  ▼
                              Class 0/1
                                  │
                                  ▼
                         Confusion Matrix
                                  │
                                  ▼
                    Accuracy / Precision
                       / Recall / F1
```

---

# 🔑 Day 4 Cheat Sheet

| Concept             | Key idea                              |
| ------------------- | ------------------------------------- |
| Classification      | Predict a class/category              |
| Logistic Regression | Classification algorithm              |
| `z`                 | `wx + b`                              |
| Sigmoid             | Converts `z` into probability         |
| Probability         | Value between 0 and 1                 |
| Threshold           | Converts probability to class         |
| Log Loss            | Measures probability prediction error |
| Gradient Descent    | Optimizes model parameters            |
| `predict()`         | Returns class                         |
| `predict_proba()`   | Returns probabilities                 |
| Decision Boundary   | Separates classes                     |
| TP                  | Correct positive                      |
| TN                  | Correct negative                      |
| FP                  | False alarm                           |
| FN                  | Missed positive                       |
| Accuracy            | Overall correctness                   |
| Precision           | Correctness of positive predictions   |
| Recall              | Positive cases successfully found     |
| F1                  | Balance of precision and recall       |

---

# Example Applications to Apply Logistic Regression and Classifications 
This same workflow can later be applied to problems such as:

```text
Spam Detection
Fraud Detection
Customer Churn
Disease Classification
Loan Default Prediction
```

The **data and problem change**, but the fundamental ML workflow remains similar.
