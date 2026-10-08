# Day 7 — Random Forest 🌲

## 📌 Overview

Random Forest is a **supervised ensemble learning algorithm** that combines multiple Decision Trees to make more stable and robust predictions.

Instead of depending on one Decision Tree, Random Forest creates many trees and combines their predictions.

```text
                Dataset
                   ↓
       ┌───────────┼───────────┐
       ↓           ↓           ↓
    Tree 1       Tree 2      Tree 3
       ↓           ↓           ↓
      Yes          No          Yes
       ↓           ↓           ↓
       └───────────┼───────────┘
                   ↓
                Voting
                   ↓
            Final Prediction
```

---

# 1. Why Random Forest?

A single Decision Tree can easily **overfit** the training data.

For example:

```text
Training Accuracy = 99%
Testing Accuracy  = 75%
```

The tree may have learned the training data too specifically.

Random Forest reduces this problem by combining many different Decision Trees.

### Main idea

> **Many different Decision Trees working together can produce a more stable prediction than a single Decision Tree.**

---

# 2. Ensemble Learning

**Ensemble Learning** means combining multiple models to produce a stronger model.

Random Forest is an ensemble of Decision Trees.

```text
Tree 1 → Prediction
Tree 2 → Prediction
Tree 3 → Prediction
Tree 4 → Prediction
...
Tree 100 → Prediction
             ↓
       Combine Results
             ↓
      Final Prediction
```

---

# 3. How Random Forest Works

Random Forest introduces randomness in two important ways:

### ① Random Rows — Bootstrap Sampling

Each Decision Tree receives a different bootstrap sample of the training data.

```text
            Dataset
               ↓
        ┌──────┬──────┬
        ↓      ↓      ↓
        Tree 1 Tree 2 Tree 3
        ↓      ↓      ↓
        Different samples of rows
```

A row can appear more than once in a bootstrap sample, while some rows may not appear in that particular sample.

---

### ② Random Features

At a split, each tree considers a random subset of available features.

For example:

```text
Tree 1 → tenure, Contract, MonthlyCharges
Tree 2 → TotalCharges, PaymentMethod, InternetService
Tree 3 → tenure, SeniorCitizen, Contract
```

This makes the trees different from one another.

---

# 4. Classification — Majority Voting

For classification, Random Forest combines the predictions using **majority voting**.

Example:

```text
Tree 1 → Churn
Tree 2 → Stay
Tree 3 → Churn
Tree 4 → Churn
Tree 5 → Stay
```

Votes:

```text
Churn → 3
Stay  → 2
```

Therefore:

```text
Final Prediction → Churn
```

---

# 5. Regression — Averaging

Random Forest can also be used for regression.

Example:

```text
Tree 1 → 50
Tree 2 → 55
Tree 3 → 52
Tree 4 → 48
Tree 5 → 51
```

The forest combines the predictions using their average.

```text
Final Prediction ≈ 51.2
```

Therefore:

| Problem        | Combination     |
| -------------- | --------------- |
| Classification | Majority voting |
| Regression     | Averaging       |

---

# 6. Decision Tree vs Random Forest

| Decision Tree       | Random Forest                    |
| ------------------- | -------------------------------- |
| One tree            | Many trees                       |
| Can overfit easily  | Usually more robust              |
| Simple to visualize | Harder to visualize as one model |
| Faster              | Usually slower                   |
| One set of rules    | Many sets of rules               |
| Single model        | Ensemble of models               |

> Random Forest can still overfit. It is not completely immune to overfitting.

---

# 7. Random Forest with Scikit-Learn

Import:

```python
from sklearn.ensemble import RandomForestClassifier
```

Create the model:

```python
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```

Train:

```python
model.fit(X_train, y_train)
```

Predict:

```python
y_pred = model.predict(X_test)
```

Get probabilities:

```python
probabilities = model.predict_proba(X_test)
```

---

# 8. Simple Random Forest Example

### Check Here
 <a href="day7-Random Forest/Random-Forest.py">📂 View Example Code</a>


### Dataset meaning

```text
Study Hours → Feature (X)

0 → Fail
1 → Pass
```

The model learns the relationship between study hours and the result.

---

# 9. `predict()` vs `predict_proba()`

### `predict()`

Returns the predicted class.

```python
model.predict([[4.5]])
```

Example:

```text
[0]
```

Meaning:

```text
0 → Fail
```

---

### `predict_proba()`

Returns the estimated probability for each class.

Example:

```python
model.predict_proba([[4.5]])
```

Output:

```text
[[0.76 0.24]]
```

The columns correspond to:

```text
Class 0 → 0.76
Class 1 → 0.24
```

Therefore:

```text
Fail → 76%
Pass → 24%

Prediction → Fail
```

> These are model probability estimates, not guaranteed real-world probabilities.

---

# 10. Important Random Forest Hyperparameters

## `n_estimators`

Number of Decision Trees.

```python
n_estimators=100
```

means approximately:

```text
100 Decision Trees
```

Examples:

```python
n_estimators=10
n_estimators=100
n_estimators=500
```

More trees can make predictions more stable but require more computation.

---

## `max_depth`

Maximum depth of each Decision Tree.

```python
max_depth=5
```

Smaller depth:

```text
Simpler trees
↓
Less complexity
↓
Can reduce overfitting
```

Larger depth:

```text
More complex trees
↓
Can learn more patterns
↓
Can potentially overfit
```

---

## `min_samples_split`

Minimum number of samples required to split a node.

```python
min_samples_split=2
```

For example:

```python
min_samples_split=10
```

requires at least 10 samples before a node can be split.

---

## `min_samples_leaf`

Minimum number of samples required in a leaf.

```python
min_samples_leaf=1
```

Increasing this value can make the trees simpler.

Example:

```python
min_samples_leaf=5
```

---

## `max_features`

Controls how many features are considered when searching for a split.

Example:

```python
max_features="sqrt"
```

This helps create more diverse trees.

---

## `random_state`

Controls randomness and makes experiments reproducible.

```python
random_state=42
```

`42` is not special; another integer can also be used.

---

# 11. Hyperparameter Cheat Sheet

| Hyperparameter      | Purpose                            |
| ------------------- | ---------------------------------- |
| `n_estimators`      | Number of trees                    |
| `max_depth`         | Maximum tree depth                 |
| `min_samples_split` | Minimum samples required to split  |
| `min_samples_leaf`  | Minimum samples required in a leaf |
| `max_features`      | Features considered at a split     |
| `random_state`      | Reproducibility                    |

---

### Important

Feature importance does **not** mean that a feature causes the target.

```text
Feature Importance ≠ Causation
```

It describes how useful the feature was to the model's decision-making process.

---

# 12. Decision Tree vs Random Forest Experiment

Both models should be trained using the same training data and evaluated using the same test data.

### Check Here 
<a href="day7-Random Forest/P1-Decision Tree vs Random Forest.py">📂 View Decision Tree Vs Random Forest Code</a>

```text
Training Accuracy
        ↓
How well did the model learn known data?

Testing Accuracy
        ↓
How well does the model generalize to unseen data?
```

A large difference can indicate overfitting.


---

# 13. Important limitation of the simple example

Our example had only:

```text
8 samples
```

and only:

```text
1 feature
```

So if we get:

```text
Accuracy = 1.0
```

that does **not** mean the model is 100% accurate in real-world situations.

For example:

```text
2 correct predictions
────────────────────── = 100%
2 test samples
```

A larger dataset provides a much more meaningful evaluation.

---

# 🧠 Day 7 — Key Points

### Remember these 7 points:

1. **Random Forest = many Decision Trees.**
2. It is an **ensemble learning** algorithm.
3. It uses **bootstrap samples of rows**.
4. It uses **random subsets of features**.
5. Classification uses **majority voting**.
6. Regression uses **averaging**.
7. Random Forest is generally more robust than a single Decision Tree, but it can still overfit.

### Mental model

```text
                 Random Forest
                      │
                      ↓
             Many Decision Trees
                      │
          ┌───────────┴───────────┐
          ↓                       ↓
   Random row samples      Random features
          ↓                       ↓
      Different trees       Different trees
          └───────────┬───────────┘
                      ↓
                Combine results
                      ↓
          ┌───────────┴───────────┐
          ↓                       ↓
    Classification           Regression
          ↓                       ↓
     Majority vote             Average
          ↓                       ↓
       Prediction              Prediction
```

## ✅ Day 7

| Topic                          | 
| ------------------------------ | 
| Random Forest concept          | 
| Ensemble Learning              |
| Bootstrap Sampling             | 
| Random Feature Selection       |
| Why Random Forest              | 
| Classification Voting          | 
| Regression Averaging           | 
| `RandomForestClassifier`       |
| `fit()`                        | 
| `predict()`                    | 
| `predict_proba()`              | 
| `n_estimators`                 | 
| `max_depth`                    | 
| `min_samples_split`            | 
| `min_samples_leaf`             | 
| `max_features`                 | 
| `random_state`                 | 
| Feature Importance             | 
| Model Evaluation               | 
| Decision Tree vs Random Forest | 
