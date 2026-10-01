import numpy as np

X = np.array([1,2,3,4,5,6])
y = np.array([35, 42, 50, 58, 65, 72])

n= len(X)

w,b = 0,0
learning_rate = 0.01

epochs = 1000

losses =[]

for epoch in range(epochs):
    y_pred = w*X+b

    error = y - y_pred
    mse = np.mean(error**2)
    losses.append(mse)

    dw = -(2/n)* np.sum(X*error)
    db = -(2/n)* np.sum(error)

    w = w - learning_rate*dw
    b = b - learning_rate*db

print("w:",w)
print("b:",b)
print("Loss MSE:",losses)

import matplotlib.pyplot as plt

plt.plot(losses)

plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Gradient Descent — Loss vs Epoch")

plt.show()