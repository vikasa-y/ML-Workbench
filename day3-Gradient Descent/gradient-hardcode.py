import numpy as np

X = np.array([1,2,3,4,5])
y = np.array([2,4,6,8,10])

n = len(X)
w , b = 0,0

learning_rate = 0.01
losses = []

for epoch in range(1000):
    y_pred = w*X + b

    error = y - y_pred

    mse = np.mean(error**2)
    losses.append(mse)

    dw = -(2/n) * np.sum(X*error)
    db = -(2/n) * np.sum(error)

    w = w - learning_rate*dw
    b = b - learning_rate*db

print("W :",w)
print("b :",b)
print("Final MSE :",losses[-1])

# PlOT LOSS vs Epoch

import matplotlib.pyplot as plt

plt.plot(losses)

plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Gradient Descent — Loss vs Epoch")

plt.show()