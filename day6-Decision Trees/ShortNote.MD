# Day 6 — Decision Trees 🌳

## 📌 Overview

A **Decision Tree** is a supervised machine learning algorithm used for both:

* Classification
* Regression

A Decision Tree learns a series of **if-else decision rules** from training data and uses those rules to make predictions.

Example:

```text
             Study Hours
               > 4.5?
              /       \
            No         Yes
            ↓           ↓
          Fail         Pass
```

---

# 1. How a Decision Tree Works

A Decision Tree repeatedly divides the dataset into smaller groups using feature-based conditions.

General process:

```text
Dataset
   ↓
Try possible splits
   ↓
Measure split quality
   ↓
Choose the best split
   ↓
Split the data
   ↓
Repeat
   ↓
Leaf Node
   ↓
Prediction
```

The tree tries to create groups that are as **pure** as possible.

---

# 2. Main Components of a Decision Tree

| Component     | Meaning                          |
| ------------- | -------------------------------- |
| Root Node     | First/main decision in the tree  |
| Internal Node | A decision inside the tree       |
| Branch        | Result of a decision             |
| Leaf Node     | Final prediction                 |
| Split         | Dividing data based on a feature |

Example:

```text
              Contract?
             /         \
     Month-to-month   Two-year
          ↓              ↓
        Churn           Stay
```

---

# 3. Example Dataset

Suppose we want to predict whether a student passes based on study hours.

| Study Hours | Result |
| ----------: | ------ |
|           1 | Fail   |
|           2 | Fail   |
|           3 | Fail   |
|           4 | Fail   |
|           5 | Pass   |
|           6 | Pass   |
|           7 | Pass   |
|           8 | Pass   |

The Decision Tree may learn a rule similar to:

```text
IF Study Hours <= 4.5
    → Fail
ELSE
    → Pass
```

The important point is that **we don't manually write this rule**.

The Decision Tree learns it from the training data.

---

# 4. How Does the Tree Choose a Split?

Suppose the tree has to decide between:

```text
Study Hours <= 3.5
Study Hours <= 4.5
Study Hours <= 5.5
```

It evaluates each possible split.

The tree asks:

> Which split creates the purest child groups?

To answer this, Decision Trees use **impurity measures**.

The two important measures are:

1. Gini Impurity
2. Entropy

---

# 5. Gini Impurity

**Gini Impurity** measures how mixed the classes are inside a node.

### Formula

$$
Gini = 1 - \sum p_i^2
$$

For binary classification:

$$
Gini = 1 - (p_0^2 + p_1^2)
$$

Where:

* \(p_0\) = proportion of class 0
* \(p_1\) = proportion of class 1

---

## Pure Node

Suppose a node contains:

```text
10 Fail
0 Pass
```

Then:

```text
P(Fail) = 1
P(Pass) = 0
```

Therefore:

$$
Gini = 1 - (1^2 + 0^2)
$$

$$
Gini = 0
$$

So:

```text
Gini = 0 → Completely pure
```

---

## 50/50 Node

Suppose:

```text
5 Fail
5 Pass
```

Then:

```text
P(Fail) = 0.5
P(Pass) = 0.5
```

Therefore:

$$
Gini = 1 - (0.5^2 + 0.5^2)
$$

$$
Gini = 1 - 0.5
$$

$$
Gini = 0.5
$$

For binary classification:

```text
Gini = 0     → Pure
Gini = 0.5   → Maximum impurity
```

---

# 6. Entropy

**Entropy** is another way to measure impurity or disorder in a node.

### Formula

$$
Entropy = -\sum p_i\log_2(p_i)
$$

For binary classification:

$$
Entropy =
-(p_0\log_2p_0 + p_1\log_2p_1)
$$

---

## Pure Node

Suppose:

```text
10 Fail
0 Pass
```

Then:

```text
P(Fail) = 1
P(Pass) = 0
```

Therefore:

$$
Entropy = 0
$$

So:

```text
Entropy = 0 → Completely pure
```

---

## 50/50 Node

Suppose:

```text
5 Fail
5 Pass
```

Then:

```text
P(Fail) = 0.5
P(Pass) = 0.5
```

Therefore:

$$
Entropy =
-(0.5\log_2 0.5 + 0.5\log_2 0.5)
$$

$$
Entropy = 1
$$

For binary classification:

```text
Entropy = 0 → Pure
Entropy = 1 → Maximum impurity
```

---

# 7. Gini vs Entropy

Both measure **impurity**, but they use different mathematical formulas.

| Feature                  | Gini                   | Entropy                               |
| ------------------------ | ---------------------- | ------------------------------------- |
| Measures                 | Impurity               | Impurity / disorder                   |
| Pure node                | 0                      | 0                                     |
| Binary maximum           | 0.5                    | 1                                     |
| Formula                  | \(1-\sum p_i^2\)       | \(-\sum p_i\log_2p_i\)                |
| Common sklearn criterion | `"gini"`               | `"entropy"`                           |
| Main idea                | How mixed is the node? | How uncertain/disordered is the node? |

Both can produce very similar Decision Trees.

---

# 8. Information Gain

Entropy tells us how impure a node is.

But the Decision Tree needs to determine:

> Which split is better?

This is where **Information Gain** is used.

### Formula

$$
Information\ Gain =
Entropy(parent) -
Weighted\ Entropy(children)
$$

A good split has **high Information Gain**.

---

## Example

Suppose the parent node contains:

```text
5 Fail
5 Pass
```

The parent is completely mixed.

```text
Entropy = 1
```

Now suppose a split creates:

```text
Left Child:
5 Fail
0 Pass

Right Child:
0 Fail
5 Pass
```

Both children are pure:

```text
Left Entropy  = 0
Right Entropy = 0
```

Weighted child entropy:

```text
0
```

Therefore:

$$
Information\ Gain = 1 - 0
$$

$$
Information\ Gain = 1
$$

This is an excellent split.

---

# 9. Weighted Child Impurity

Children may not contain the same number of samples.

Example:

```text
Parent = 10 samples

Left Child  = 7 samples
Right Child = 3 samples
```

The child impurity is weighted according to the number of samples.

$$
Weighted\ Impurity =
\frac{7}{10}Impurity_{left}
+
\frac{3}{10}Impurity_{right}
$$

This prevents a tiny child node from having the same influence as a much larger child node.

---

# 10. How the Tree Selects the Best Split

Conceptually:

```text
                    Dataset
                       ↓
             Try possible splits
                       ↓
              Calculate impurity
                       ↓
               Compare splits
                       ↓
               Choose best split
                       ↓
                  Split data
                       ↓
              Repeat recursively
                       ↓
                  Leaf Node
```

For **Gini**, the tree generally prefers the split with the **lowest weighted Gini impurity**.

For **Entropy**, the tree generally prefers the split with the **highest Information Gain**.

---

# 11. `criterion` in Scikit-Learn

The `criterion` parameter tells the Decision Tree how to evaluate the quality of a split.

### Gini

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    random_state=42
)
```

`gini` is the default criterion for `DecisionTreeClassifier`.

---

### Entropy

```python
model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=3,
    random_state=42
)
```

This uses entropy-based splitting.

---

### Log Loss

Scikit-learn also supports:

```python
model = DecisionTreeClassifier(
    criterion="log_loss",
    max_depth=3,
    random_state=42
)
```

For now, remember:

```text
criterion="gini"
        ↓
Gini impurity

criterion="entropy"
        ↓
Entropy / Information Gain

criterion="log_loss"
        ↓
Log-loss-based splitting
```

---

# 12. Decision Tree with Scikit-Learn

```python
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
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

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print(
    "Classification Report:",
    classification_report(y_test, y_pred)
)

print(
    "Confusion Matrix:",
    confusion_matrix(y_test, y_pred)
)
```

---

# 13. `predict()` and `predict_proba()`

A Decision Tree can provide both a predicted class and class probabilities.

### Prediction

```python
model.predict(X_test)
```

Example:

```text
[0, 1]
```

---

### Probability

```python
model.predict_proba(X_test)
```

Example:

```text
[
    [0.9, 0.1],
    [0.1, 0.9]
]
```

The columns represent:

```text
[Probability of Class 0, Probability of Class 1]
```

So:

```text
[0.9, 0.1]
```

means:

```text
90% → Fail
10% → Pass
```

and:

```text
[0.1, 0.9]
```

means:

```text
10% → Fail
90% → Pass
```

---

# 14. Visualizing the Decision Tree

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plot_tree(
    model,
    feature_names=["Study Hours"],
    class_names=["Fail", "Pass"],
    filled=True
)

plt.show()
```

The visualization allows us to see the actual decision rules learned by the model.

---

# 15. Overfitting in Decision Trees

Decision Trees can easily **overfit**.

A tree can continue splitting until it learns very specific patterns from the training data.

Example:

```text
Training Accuracy → 100%
Testing Accuracy  → Poor
```

This means the tree may have memorized the training data instead of learning general patterns.

---

# 16. Controlling Overfitting

Important Decision Tree hyperparameters:

### `max_depth`

Controls the maximum depth of the tree.

```python
DecisionTreeClassifier(max_depth=3)
```

Smaller depth:

```text
Simpler tree
↓
Less overfitting
↓
Possible underfitting
```

Larger depth:

```text
More complex tree
↓
Can capture more patterns
↓
Higher risk of overfitting
```

---

### `min_samples_split`

Minimum number of samples required to split an internal node.

```python
DecisionTreeClassifier(
    min_samples_split=5
)
```

---

### `min_samples_leaf`

Minimum number of samples that must be present in a leaf node.

```python
DecisionTreeClassifier(
    min_samples_leaf=2
)
```

---

### Important Hyperparameters

| Hyperparameter      | Purpose                                  |
| ------------------- | ---------------------------------------- |
| `criterion`         | Measures split quality                   |
| `max_depth`         | Limits tree depth                        |
| `min_samples_split` | Minimum samples required to split        |
| `min_samples_leaf`  | Minimum samples in a leaf                |
| `max_features`      | Limits features considered for splitting |

---

# 17. Decision Tree vs Previous Algorithms

| Algorithm           | Main Idea                                |
| ------------------- | ---------------------------------------- |
| Linear Regression   | Fits a line to predict numerical values  |
| Logistic Regression | Uses probability and a decision boundary |
| KNN                 | Looks at nearby training examples        |
| Naive Bayes         | Uses probability and Bayes' theorem      |
| Decision Tree       | Learns if-else decision rules            |

---

# 18. Advantages of Decision Trees

### Advantages

* Easy to understand
* Easy to visualize
* Can handle classification and regression
* Can capture non-linear relationships
* Does not require feature scaling
* Produces human-readable decision rules
* Works with numerical and categorical features after appropriate preprocessing

---

# 19. Disadvantages of Decision Trees

### Disadvantages

* Can easily overfit
* Small changes in data can change the tree
* Deep trees can become very complex
* A single tree may have lower generalization performance than ensemble methods
* Usually benefits from pruning or controlling tree complexity

This is one reason algorithms such as **Random Forest** and **Gradient Boosting** are important later.

---

# 20. Decision Tree Mental Model

Think of a Decision Tree as an automated question-answer system:

```text
                 Feature?
                    ↓
              Ask a question
                    ↓
              ┌─────┴─────┐
             Yes           No
              ↓             ↓
          Ask another    Ask another
           question       question
              ↓             ↓
             ...           ...
              ↓             ↓
             Leaf          Leaf
              ↓             ↓
          Prediction    Prediction
```

The important difference is:

> **We don't manually create the questions. The algorithm learns the best questions from the data.**

---

# 21. Key Formulas

### Gini Impurity

$$
\boxed{Gini = 1-\sum p_i^2}
$$

### Entropy

$$
\boxed{Entropy=-\sum p_i\log_2(p_i)}
$$

### Information Gain

$$
\boxed{
Information\ Gain =
Entropy(parent)-Weighted\ Entropy(children)
}
$$

---

# 22. Quick Revision

```text
Decision Tree
      ↓
Learns if/else rules
      ↓
Splits dataset
      ↓
Measures impurity
      ↓
Gini / Entropy
      ↓
Chooses good split
      ↓
Repeats recursively
      ↓
Leaf Node
      ↓
Prediction
```

### Remember

```text
Gini
↓
Measures impurity
↓
0 = pure

Entropy
↓
Measures impurity/disorder
↓
0 = pure

Information Gain
↓
Measures improvement after a split
↓
Higher = better split

max_depth
↓
Controls tree complexity
↓
Helps prevent overfitting
```

---


