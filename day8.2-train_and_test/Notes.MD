# Train-Test Split & Model Training

## 📌 Overview

Before training a machine learning model, we usually divide the dataset into **training data** and **testing data**.

The main purpose is to answer:

> **Can the model make good predictions on data it has never seen before?**

A model should not only memorize the training data. It should learn patterns that **generalize** to new data.

---

# 1. Why Do We Split the Dataset?

Suppose we have:

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

If we train the model using all the data and then test it on the **same data**, the model has already seen the answers.

For example:

```text
Training Data
     ↓
   Model
     ↓
Same Training Data
     ↓
Evaluation
```

A very high accuracy in this situation does not necessarily mean the model can handle new data.

Instead, we separate the data:

```text
                Dataset
                   ↓
            Train/Test Split
              ↙         ↘
        Training        Testing
           ↓               ↓
      Learn patterns   Check performance
```

---

# 2. Training Data

**Training data** is the data used by the model to learn patterns.

It contains:

```text
X_train → Features / Inputs
y_train → Correct Target / Answers
```

Example:

```text
X_train          y_train

Study Hours      Result
1                Fail
2                Fail
3                Fail
5                Pass
6                Pass
7                Pass
```

The model learns the relationship between the features and the target.

For a Decision Tree, it may learn a rule such as:

```text
Study Hours <= 4.5
        ↓
      Fail

Study Hours > 4.5
        ↓
      Pass
```

The important point:

> **Training data teaches the model.**

---

# 3. Testing Data

**Testing data** is data that is kept separate from training.

It is used to check whether the trained model can make predictions on **unseen data**.

It contains:

```text
X_test → Test features
y_test → Actual test answers
```

Example:

```text
X_test          y_test

Study Hours     Result
4               Fail
8               Pass
```

The model receives only:

```text
X_test
```

and produces:

```text
y_pred
```

Then we compare:

```text
y_test  → Actual answers
y_pred  → Model predictions
```

The important point:

> **Testing data checks the model's ability to generalize.**

---

# 4. Training vs Testing

|                                    | Training Data  | Testing Data   |
| ---------------------------------- | -------------- | -------------- |
| Purpose                            | Learn patterns | Evaluate model |
| Features                           | `X_train`      | `X_test`       |
| Target                             | `y_train`      | `y_test`       |
| Model sees target during training? | Yes            | No             |
| Used in `.fit()`?                  | Yes            | No             |
| Used in `.predict()`?              | No             | Yes            |
| Used for final evaluation?         | No             | Yes            |

---

# 5. `train_test_split()`

Scikit-learn provides:

```python
from sklearn.model_selection import train_test_split
```

We can split our data using:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

---

# 6. Understanding Each Parameter

## `X`

`X` contains the input features.

Example:

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
```

---

## `y`

`y` contains the target/output.

```python
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
```

Here:

```text
0 → Fail
1 → Pass
```

---

# 7. `test_size`

```python
test_size=0.2
```

means approximately:

```text
20% → Testing
80% → Training
```

For a dataset containing 1,000 samples:

```text
800 → Training
200 → Testing
```

Common choices include:

```text
test_size=0.2 → 80/20 split
test_size=0.3 → 70/30 split
test_size=0.25 → 75/25 split
```

The appropriate split depends on the dataset size and problem.

---

# 8. `random_state`

```python
random_state=42
```

`train_test_split()` randomly selects which samples go into training and testing.

Without a fixed `random_state`, the split can change between runs.

With:

```python
random_state=42
```

the same split can be reproduced.

Example:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### Important

`42` has no special mathematical meaning.

You could use:

```python
random_state=10
```

or:

```python
random_state=100
```

The important idea is **reproducibility**.

---

# 9. The Four Resulting Variables

After:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

we have four datasets.

### `X_train`

Training features.

```text
Inputs used for learning
```

### `y_train`

Training targets.

```text
Correct answers used during learning
```

### `X_test`

Testing features.

```text
Unseen inputs
```

### `y_test`

Testing targets.

```text
Correct answers used for evaluation
```

---

# 10. Model Training

Once the data is split, we create a model.

Example:

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)
```

At this point, the model has **not learned from the dataset yet**.

We train it using:

```python
model.fit(X_train, y_train)
```

---

# 11. What Does `.fit()` Mean?

```python
model.fit(X_train, y_train)
```

means:

> **Learn the relationship between the training features and training targets.**

Conceptually:

```text
X_train + y_train
       ↓
    model.fit()
       ↓
  Model learns
       ↓
 Trained Model
```

Different algorithms learn different things.

For example:

```text
Linear Regression
→ learns coefficients

Logistic Regression
→ learns coefficients for classification

KNN
→ stores training examples

Decision Tree
→ learns decision rules/splits

Naive Bayes
→ learns probability distributions
```

---

# 12. Making Predictions

After training:

```python
y_pred = model.predict(X_test)
```

The model has never used `X_test` for training.

Therefore:

```text
X_test
   ↓
Trained Model
   ↓
y_pred
```

`y_pred` contains the model's predictions.

---

# 13. Evaluating Predictions

Now we have:

```text
y_test → Actual answers
y_pred → Model predictions
```

We compare them.

For example:

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
```

Conceptually:

```text
Actual       Predicted
  ↓              ↓
y_test         y_pred
     \          /
      \        /
       Compare
          ↓
      Evaluation
```

---

# 14. Complete ML Flow

The basic supervised ML workflow is:

```text
                 DATASET
                    ↓
              X and y
                    ↓
            Train/Test Split
                    ↓
          ┌─────────┴─────────┐
          ↓                   ↓
      Training              Testing
          ↓                   ↓
X_train + y_train         X_test + y_test
          ↓
      model.fit()
          ↓
      Model learns
          ↓
      X_test
          ↓
   model.predict()
          ↓
        y_pred
          ↓
Compare y_pred with y_test
          ↓
      Evaluation
```

---

# 15. Why Not Train and Test on the Same Data?

Bad workflow:

```python
model.fit(X, y)

y_pred = model.predict(X)

accuracy_score(y, y_pred)
```

Here, the model is evaluated on the same data it already learned from.

This can give an overly optimistic result.

Correct workflow:

```python
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy_score(y_test, y_pred)
```

Now the model is evaluated on data it did not use during training.

---

# 16. Generalization

One of the main goals of machine learning is **generalization**.

### Generalization

The ability of a model to perform well on **new, unseen data**.

```text
Training Data
     ↓
Learn patterns
     ↓
Model
     ↓
Unseen Data
     ↓
Good Predictions
```

A model that performs well only on training data is not necessarily a good model.

---

# 17. Training Accuracy vs Testing Accuracy

We can calculate training performance:

```python
train_pred = model.predict(X_train)

train_accuracy = accuracy_score(
    y_train,
    train_pred
)
```

And testing performance:

```python
test_pred = model.predict(X_test)

test_accuracy = accuracy_score(
    y_test,
    test_pred
)
```

Example:

```text
Training Accuracy = 100%
Testing Accuracy  = 70%
```

This can indicate **overfitting**.

The model may have learned the training data too specifically.

---

# 18. Overfitting

Overfitting occurs when a model learns the training data extremely well but performs poorly on unseen data.

```text
Training Performance → Very High
Testing Performance  → Poor
```

Conceptually:

```text
Simple Model
     ↓
May underfit

Good Generalization
     ↓
Desired

Very Complex Model
     ↓
May overfit
```

Decision Trees are particularly capable of overfitting when they grow too deep.

This is why we use parameters such as:

```python
DecisionTreeClassifier(
    max_depth=3,
    min_samples_split=5,
    min_samples_leaf=2
)
```

---

# 19. Train / Validation / Test

Later, when tuning models, we may use three datasets:

```text
Dataset
   ↓
Train + Validation + Test
```

### Training Set

Used to train the model.

### Validation Set

Used to tune hyperparameters and compare models.

### Test Set

Used for the final evaluation.

Example:

```text
70% → Training
15% → Validation
15% → Testing
```

The exact percentages can vary.

For now, the basic workflow is:

```text
Train/Test Split
       ↓
Train Model
       ↓
Predict
       ↓
Evaluate
```

Validation becomes especially important when we study **hyperparameter tuning and cross-validation**.

---

# 20. Important Rule

Never use test data to train the model.

```text
Correct:

X_train + y_train
       ↓
    .fit()
       ↓
    Model
       ↓
X_test
       ↓
 .predict()
       ↓
   y_pred
       ↓
Compare with y_test
```

Remember:

> **Training data teaches the model. Test data checks whether it learned well enough to handle unseen data.**

---

# 21. Example with Decision Tree

```python
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
```

---

# 22. Quick Revision

```text
X
↓
Features / Inputs

y
↓
Target / Correct Answers

X_train
↓
Training Features

y_train
↓
Training Targets

X_test
↓
Unseen Testing Features

y_test
↓
Actual Testing Targets

model.fit(X_train, y_train)
↓
Train / Learn

model.predict(X_test)
↓
Predict

y_pred
↓
Predicted Answers

y_test vs y_pred
↓
Evaluate
```

---

# 🎯 Key Takeaway

> **We split data so that the model learns from one portion and is evaluated on another portion it has not seen during training. This allows us to measure how well the model generalizes to unseen data.**

### Core workflow

```text
Dataset
   ↓
X, y
   ↓
Train/Test Split
   ↓
X_train + y_train
   ↓
model.fit()
   ↓
Trained Model
   ↓
X_test
   ↓
model.predict()
   ↓
y_pred
   ↓
Compare with y_test
   ↓
Evaluate Model
```

This **train → predict → evaluate** pattern is fundamental to almost every supervised ML algorithm.
