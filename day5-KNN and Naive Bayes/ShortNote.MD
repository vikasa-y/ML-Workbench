# ML Day 5 — KNN & Naive Bayes

## 1. KNN

**KNN = K-Nearest Neighbors**

KNN predicts a new data point by looking at the **K closest training examples**.

### Core idea

```text
New point
   ↓
Calculate distances
   ↓
Find K nearest points
   ↓
Look at their labels
   ↓
Majority vote
   ↓
Prediction
```

### Example

```python
X = [1, 2, 3, 7, 8, 9]
y = [0, 0, 0, 1, 1, 1]
```

Here:

```text
0 = Fail
1 = Pass
```

For:

```python
new_point = 5
K = 3
```

Nearest labels:

```text
[0, 1, 0]
```

Majority:

```text
0 → 2 votes
1 → 1 vote

Prediction = 0 → Fail
```

---

## 2. Distance

For one feature:

$$
distance = |x_1-x_2|
$$

For multiple features, commonly Euclidean distance:

$$
d=\sqrt{\sum (x_i-y_i)^2}
$$

### Important

KNN is **distance-based**, so feature scaling is important.

```text
Raw data
   ↓
StandardScaler
   ↓
KNN
```

---

## 3. KNN Scratch Code

```python
import numpy as np

X = np.array([1, 2, 3, 7, 8, 9])
y = np.array([0, 0, 0, 1, 1, 1])

new_point = 5
K = 3

distances = np.abs(X - new_point)

indices = np.argsort(distances)

nearest_indices = indices[:K]

nearest_labels = y[nearest_indices]

prediction = np.bincount(nearest_labels).argmax()

print(prediction)
```

### Important NumPy functions

| Function        | Purpose                          |
| --------------- | -------------------------------- |
| `np.abs()`      | Absolute distance                |
| `np.argsort()`  | Returns indices that sort values |
| `[:K]`          | Select K nearest                 |
| `np.bincount()` | Count class occurrences          |
| `.argmax()`     | Find index with largest count    |

---

## 4. KNN with sklearn

```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=3)

model.fit(X, y)

prediction = model.predict([[5]])
```

`n_neighbors=3` means:

```text
K = 3
```

### Important characteristic

KNN doesn't learn `w` and `b` like Linear/Logistic Regression.

It mainly stores the training data and uses distances when making predictions.

---

# 5. Naive Bayes

Naive Bayes is a **family of probabilistic classification algorithms** based on Bayes' theorem.

Bayes theorem:

$$
P(A|B)=\frac{P(B|A)P(A)}{P(B)}
$$

Core idea:

```text
Features
   ↓
Calculate class probabilities
   ↓
Compare probabilities
   ↓
Choose most probable class
```

### Why "Naive"?

It assumes features are **conditionally independent given the class**.

Simplified:

$$
P(x_1,x_2|class)
\approx
P(x_1|class)P(x_2|class)
$$

---

# 6. Types of Naive Bayes

Naive Bayes is **not only GaussianNB**.

| Type        | sklearn           | Typical use               |
| ----------- | ----------------- | ------------------------- |
| Gaussian    | `GaussianNB()`    | Continuous numerical data |
| Multinomial | `MultinomialNB()` | Counts / text             |
| Bernoulli   | `BernoulliNB()`   | Binary features           |
| Categorical | `CategoricalNB()` | Categorical features      |

### Examples

```text
Age, height, temperature
        ↓
GaussianNB
```

```text
Word counts / text
        ↓
MultinomialNB
```

```text
Feature = 0 or 1
        ↓
BernoulliNB
```

---

# 7. Gaussian Naive Bayes

```python
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()

model.fit(X, y)

prediction = model.predict([[5]])

probability = model.predict_proba([[5]])
```

`predict()` → predicted class

`predict_proba()` → probability for each class

---

# 8. KNN vs Logistic Regression vs Naive Bayes

| Algorithm           | Main idea                              |
| ------------------- | -------------------------------------- |
| Logistic Regression | Learn a mathematical decision boundary |
| KNN                 | Look at nearby examples                |
| Naive Bayes         | Calculate class probabilities          |

### Mental shortcut

```text
Logistic Regression
→ Learn boundary

KNN
→ Find neighbors

Naive Bayes
→ Calculate probabilities
```

---

# 9. Real-world thinking

The goal isn't:

> "Which algorithm is best?"

Instead:

```text
Problem
   ↓
Understand data
   ↓
Choose suitable algorithms
   ↓
Train
   ↓
Evaluate
   ↓
Compare
   ↓
Analyze errors
   ↓
Improve
```

Different algorithms can solve the same classification problem.

---

# 10. Day 5 Cheat Sheet

```text
KNN
────────────────────────
Full form → K-Nearest Neighbors
Type → Supervised
Usually → Classification
Main idea → Nearest examples
Distance → Euclidean / other distance
Important → Feature scaling
K → Number of neighbors
Prediction → Majority vote


Naive Bayes
────────────────────────
Type → Supervised
Main idea → Probability
Based on → Bayes theorem
"Naive" → Conditional independence assumption

Types:
GaussianNB       → Numerical continuous
MultinomialNB    → Counts / text
BernoulliNB      → Binary
CategoricalNB    → Categorical
```

## Final mental model

```text
                 Classification
                       │
          ┌────────────┼────────────┐
          ↓            ↓            ↓
   Logistic Reg.      KNN      Naive Bayes
          │            │            │
     Learn boundary  Neighbors   Probability
          │            │            │
          └────────────┼────────────┘
                       ↓
                   Prediction
                       ↓
                  Evaluation
```

