# 🤖 ML Day 3 — Gradient Descent

## 📌 Topics Covered

* Why Gradient Descent is needed
* MSE Loss
* Derivative intuition
* Gradient
* Why we subtract the gradient
* `dw` and `db`
* Parameter updates
* Learning Rate
* Epoch
* Gradient Descent from scratch
* Loss tracking
* Convergence
* Learning-rate experiments
* Linear Regression with multiple features
* Scratch vs Scikit-learn

---

## 1. Why Gradient Descent?

In Linear Regression:

$$
\hat{y} = wx + b
$$

The model needs to find the best values of:

* `w` → weight/slope
* `b` → bias/intercept

The goal is to find `w` and `b` that produce the **smallest error**.

We use **Gradient Descent** to automatically adjust these parameters.

---

## 2. MSE Loss

Mean Squared Error measures how far predictions are from actual values.

$$
MSE = \frac{1}{n}\sum(y-\hat{y})^2
$$

Where:

* `y` → actual value
* `ŷ` → predicted value
* `n` → number of samples

Lower MSE generally means predictions are closer to the actual values.

---

## 3. Derivative Intuition

A derivative tells us:

> How much does the loss change when a parameter changes?

For example:

$$
f(w)=w^2
$$

Derivative:

$$
\frac{df}{dw}=2w
$$

If:

```text
w = 3
gradient = 6
```

The gradient is positive, so increasing `w` increases the loss.

Gradient Descent moves in the **opposite direction**.

---

## 4. Why Do We Subtract the Gradient?

The update rule is:

$$
w_{new}=w_{old}-\alpha\frac{\partial MSE}{\partial w}
$$

$$
b_{new}=b_{old}-\alpha\frac{\partial MSE}{\partial b}
$$

We subtract because the gradient points toward the direction of **increasing loss**.

Gradient Descent wants to move toward **lower loss**.

---

## 5. Linear Regression Gradients

For:

$$
\hat y = wx+b
$$

The gradients are:

$$
dw=-\frac{2}{n}\sum x(y-\hat y)
$$

$$
db=-\frac{2}{n}\sum(y-\hat y)
$$

Then update:

$$
w=w-\alpha dw
$$

$$
b=b-\alpha db
$$

---

## 6. One Manual Update

Given:

```python
X = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

w = 0
b = 0
learning_rate = 0.01
```

Initial predictions:

```text
[0, 0, 0, 0, 0]
```

Error:

```text
[2, 4, 6, 8, 10]
```

Gradients:

```text
dw = -44
db = -12
```

Update:

```text
w = 0 - 0.01(-44)
w = 0.44

b = 0 - 0.01(-12)
b = 0.12
```

So after one update:

```text
w = 0.44
b = 0.12
```

---

## 7. Learning Rate

Learning rate controls the **step size** of each update.

```python
learning_rate = 0.01
```

### Small learning rate

```text
0.001
```

* Smaller steps
* More stable
* Can be slow

### Larger learning rate

```text
0.1
```

* Larger steps
* Can converge faster
* Can overshoot or oscillate

There is **no universal best learning rate**.

---

## 8. Epoch

An **epoch** means one complete iteration through the training data.

Example:

```python
for epoch in range(1000):
```

means Gradient Descent performs **1000 parameter updates**.

---

# 9. Linear Regression From Scratch

```python
import numpy as np

X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

n = len(X)

w = 0
b = 0

learning_rate = 0.01
epochs = 1000

for epoch in range(epochs):

    # Prediction
    y_pred = w * X + b

    # Error
    error = y - y_pred

    # Gradients
    dw = -(2/n) * np.sum(X * error)
    db = -(2/n) * np.sum(error)

    # Update
    w = w - learning_rate * dw
    b = b - learning_rate * db

print("w:", w)
print("b:", b)
```

### Flow

```text
X, y
 ↓
Prediction
 ↓
Calculate Error
 ↓
Calculate Gradients
 ↓
Update w and b
 ↓
Repeat
```

---

# 10. Tracking MSE

```python
losses = []

for epoch in range(1000):

    y_pred = w * X + b

    error = y - y_pred

    mse = np.mean(error ** 2)
    losses.append(mse)

    dw = -(2/n) * np.sum(X * error)
    db = -(2/n) * np.sum(error)

    w = w - learning_rate * dw
    b = b - learning_rate * db
```

To get the final MSE:

```python
print("Final MSE:", losses[-1])
```

---

# 11. Loss Visualization

```python
import matplotlib.pyplot as plt

plt.plot(losses)

plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Gradient Descent — Loss vs Epoch")

plt.show()
```

Expected behavior:

```text
MSE
│\
│ \
│  \
│   \
│    \____
│
└──────────── Epoch
```

The loss should generally decrease as training progresses.

---

# 12. Convergence

**Convergence** means the model parameters/loss have become approximately stable.

Example:

```text
Epoch 1       → MSE = 44
Epoch 100     → MSE = 0.8
Epoch 500     → MSE = 0.02
Epoch 1000    → MSE = 0.01
```

The model is approaching a stable solution.

> Convergence does not automatically mean the model is perfect.

---

# 13. Learning Rate Experiment

```python
def train_model(X, y, learning_rate, epochs=1000):

    n = len(X)

    w = 0
    b = 0

    losses = []

    for epoch in range(epochs):

        y_pred = w * X + b

        error = y - y_pred

        mse = np.mean(error ** 2)
        losses.append(mse)

        dw = -(2/n) * np.sum(X * error)
        db = -(2/n) * np.sum(error)

        w = w - learning_rate * dw
        b = b - learning_rate * db

    return w, b, losses
```

Experiment with:

```python
0.001
0.01
0.1
```

Compare:

* Final MSE
* Speed of convergence
* Loss curve
* Stability

---

# 14. Multiple Linear Regression

With one feature:

$$
\hat y = wx+b
$$

With multiple features:

$$
\hat y=w_1x_1+w_2x_2+b
$$

Example:

```python
X = np.array([
    [1, 50],
    [2, 60],
    [3, 70],
    [4, 80]
])
```

Here:

```text
x₁ = Study Hours
x₂ = Attendance
```

The model has two weights:

```python
model.coef_
```

Example:

```text
[w1, w2]
```

Therefore:

```python
model.coef_[0]
```

→ weight for first feature

```python
model.coef_[1]
```

→ weight for second feature

### Important rule

> Number of coefficients = Number of features.

---

# 15. Intercept `b`

The intercept can be:

```text
positive
zero
negative
```

For example:

$$
\hat y=7x-3
$$

Here:

```text
w = 7
b = -3
```

A negative `b` is completely valid.

---

# 16. Scikit-learn Linear Regression

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

model = LinearRegression()

model.fit(X, y)

y_pred = model.predict(X)

print("Weights:", model.coef_)
print("Bias:", model.intercept_)

mse = mean_squared_error(y, y_pred)

print("MSE:", mse)
```

### Important attributes

```python
model.coef_
```

→ learned weights

```python
model.coef_[0]
```

→ first feature's weight

```python
model.coef_[1]
```

→ second feature's weight

```python
model.intercept_
```

→ bias/intercept `b`

---

# 17. Scratch vs Scikit-learn

| From Scratch             | Scikit-learn                                 |
| ------------------------ | -------------------------------------------- |
| We initialize `w,b`      | Library handles parameters                   |
| We calculate gradients   | Library handles optimization                 |
| We choose learning rate  | No manual LR required for `LinearRegression` |
| We write training loop   | `fit()` handles training                     |
| We calculate predictions | `predict()`                                  |
| Good for understanding   | Good for practical ML                        |

Both are solving the same basic goal:

$$
\text{Find parameters that minimize prediction error}
$$

---

# 🧠 Day 3 Cheat Sheet

### Model

$$
\hat y=wx+b
$$

### MSE

$$
MSE=\frac{1}{n}\sum(y-\hat y)^2
$$

### Gradients

$$
dw=-\frac{2}{n}\sum x(y-\hat y)
$$

$$
db=-\frac{2}{n}\sum(y-\hat y)
$$

### Update

$$
w=w-\alpha dw
$$

$$
b=b-\alpha db
$$

### Remember

```text
Gradient → direction of increasing loss
Gradient Descent → move opposite direction
Learning Rate → step size
Epoch → one training iteration
Convergence → parameters/loss become stable
w → weight/slope
b → bias/intercept
coef_ → learned weights
intercept_ → learned bias
```

---

## 🔥 Day 3 Mental Model

```text
Start with random/initial w,b
             ↓
        Make prediction
             ↓
       Calculate error
             ↓
          Calculate
          gradients
             ↓
       Update w and b
             ↓
        Repeat epochs
             ↓
       Loss decreases
             ↓
         Convergence
```

### Final Goal

Understand this sentence:

> **Gradient Descent repeatedly changes the model's parameters in the direction that reduces the loss.**
