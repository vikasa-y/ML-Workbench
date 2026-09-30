# Linear Regression + Mathematics

## 1. What is Linear Regression?
Linear Regression is a supervised machine learning algorithm used to predict a continuous numerical value. It learns a
relationship between input feature(s) X and a numerical target y.

## 2. Linear Regression Equation
n = wx + b
x = input feature n = prediction w = weight/slope b = bias/intercept

## 3. Understanding w and b
w controls the slope/steepness. b shifts the line up or down without changing the slope.
Example: w=2,b=0 gives n=2x. For x=[1,2,3,4,5], predictions are [2,4,6,8,10].
With w=2,b=2, predictions are [4,6,8,10,12].

## 4. Why Learn w and b?
In real ML, the best w and b are unknown. The model adjusts these parameters so predictions become closer to the actual
target values. The goal is to minimize prediction error.

## 5. Prediction and Error
Prediction: n = wx + b
Error/residual: y − n
y is the actual value; n is the model prediction.

## 6. Cost Function — MSE
Mean Squared Error measures the average squared difference between actual values and predictions.

## 7. Gradient Descent — Intuition
Imagine MSE as a landscape with a valley. The gradient tells the direction in which cost increases. Gradient Descent
moves in the opposite direction to reduce cost.
Loop: start w,b → predict → calculate error/MSE → calculate gradients → update w,b → repeat.

## 8. Derivatives and Gradients
A derivative tells how a quantity changes when a variable changes. ∂MSE/∂w tells how MSE changes when w changes;
∂MSE/∂b tells how MSE changes when b changes.
## 9. Learning Rate
The learning rate α controls the size of each update step. Very small can be slow; a reasonable value can converge
efficiently; too large can overshoot or become unstable. It is not a universal fixed value.
## 10. Gradient Descent Mathematics
MSE = (1/n) Σ(yn − nn)²
nn = wxn + b
∂MSE/∂w = −(2/n) Σ xn(yn − nn)
∂MSE/∂b = −(2/n) Σ(yn − nn)
w ← w − α(∂MSE/∂w)
b ← b − α(∂MSE/∂b)
## 11. Linear Regression From Scratch
Start with w=b=0. Each epoch makes predictions, calculates errors, calculates dw and db, updates w and b, and repeats.
## 12. Scikit-learn Linear Regression
fit(X,y) learns parameters. predict(X) generates predictions. coef_ contains learned weight(s). intercept_ contains
learned bias.
1## 3. coef_, [0], intercept_
model.coef_ is an array of learned coefficients. model.coef_[0] means the first coefficient; [0] is normal array indexing.
model.intercept_ is the learned bias/intercept. With multiple features, coef_ can contain one weight per feature.
## 14. Why sklearn X is 2D
Scikit-learn represents X as (number of samples, number of features). Five samples and one feature means shape (5,1).
Rows are samples; columns are features.
## 15. Visualization
Scatter points show actual observations; the regression line shows model predictions. A real fitted line does not necessarily
pass through every point.
## 16. Study Hours → Marks Mini Exercise
Use X=[[1],[2],[3],[4],[5],[6]] and y=[35,42,50,58,65,72]. Fit LinearRegression, predict, calculate MSE, and plot the actual
points with the regression line.
Worked MSE Example
X=[1,2,3,4,5], y=[2,4,6,8,10], w=1,b=0 → predictions [1,2,3,4,5]. Squared errors=[1,4,9,16,25]. Sum=55, so MSE=55/5=11.
With w=2,b=0, predictions match exactly and MSE=0.

## From-Scratch Code
```
import numpy as np

X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])
n = len(X)
w = 0
b = 0
learning_rate = 0.01

for epoch in range(1000):
y_pred = w * X + b
error = y - y_pred
a
dw = -(2/n) * np.sum(X * error)
db = -(2/n) * np.sum(error)

w = w - learning_rate * dw
b = b - learning_rate * db

print(w)
print(b)


```
## Scikit-learn Code
```
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])
model = LinearRegression()

model.fit(X, y)
y_pred = model.predict(X)

print("w:", model.coef_[0])
print("b:", model.intercept_)
print("Predciton:", y_pred)


```
| Feature | From-Scratch Code | Scikit-learn Code |
|---|---|---|
| Library | `numpy` | `numpy` + `sklearn` |
| Model | Manually implemented | `LinearRegression()` |
| Input `X` | `np.array([1, 2, 3, 4, 5])` | `np.array([[1], [2], [3], [4], [5]])` |
| Target `y` | `np.array([2, 4, 6, 8, 10])` | `np.array([2, 4, 6, 8, 10])` |
| Weight `w` | Manually initialized: `w = 0` | Learned by `model.fit()` |
| Bias `b` | Manually initialized: `b = 0` | Learned by `model.fit()` |
| Learning Rate | `learning_rate = 0.01` | Handled internally |
| Prediction | `y_pred = w * X + b` | `model.predict(X)` |
| Error | `error = y - y_pred` | Handled internally |
| Gradient `dw` | Calculated manually | Handled internally |
| Gradient `db` | Calculated manually | Handled internally |
| Parameter Update | `w = w - learning_rate * dw` | Handled internally |
| Training | `for epoch in range(1000)` | `model.fit(X, y)` |
| Weight Access | `w` | `model.coef_[0]` |
| Bias Access | `b` | `model.intercept_` |
| Prediction Output | Approximately `[2, 4, 6, 8, 10]` | `[2, 4, 6, 8, 10]` |
| Main Purpose | Understand the internal working | Build models quickly and efficiently |