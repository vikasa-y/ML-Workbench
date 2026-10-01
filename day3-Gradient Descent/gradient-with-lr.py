import numpy as np
import matplotlib.pyplot as plt

X = np.array([1, 2, 3, 4, 5, 6])
y = np.array([35, 42, 50, 58, 65, 72])

n = len(X)

learning_rates = [0.001, 0.01, 0.1]

for learning_rate in learning_rates:

    w = 0
    b = 0

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

    print("Learning Rate:", learning_rate)
    print("w:", w)
    print("b:", b)
    print("Final MSE:", losses[-1])
    print()

    plt.plot(losses, label=f"LR = {learning_rate}")

plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Learning Rate Comparison")
plt.legend()
plt.show()