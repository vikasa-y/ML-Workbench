# LR (Linear Regression)
import numpy as np

X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

n=len(X)
w=0
b=0
learning_rate = 0.01

for epoch in range(1000):
    y_pred = w*X + b

    error = y - y_pred

    dw = -(2/n)*np.sum(X*error)
    db = -(2/n)*np.sum(error)

    w = w - learning_rate * dw
    b = b - learning_rate * db

print("w:", w)
print("b:", b)
print("prediction:", w * X + b)