import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

X = np.array([
    [0, 0], [0.2, 0.1], [-0.2, 0.1],
    [0.1, -0.2], [-0.1, -0.2],
    [2, 2], [2, -2], [-2, 2], [-2, -2],
    [2, 0], [-2, 0], [0, 2], [0, -2]
])

y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])

for kernel in ["linear", "rbf"]:
    model = SVC(kernel=kernel, C=1)
    model.fit(X, y)

    predictions = model.predict(X)

    print("Kernel:", kernel)
    print("Predictions:", predictions)
    print("Training accuracy:", accuracy_score(y, predictions))
    print()


for c_value in [0.01, 1, 100]:
    model = SVC(kernel="rbf", C=c_value)
    model.fit(X, y)

    print("C:", c_value)
    print("Accuracy:", model.score(X, y))


import matplotlib.pyplot as plt

for c_value in [0.01 , 1 , 100]:
    model = SVC(kernel="rbf" , C=c_value , gamma="scale")
    model.fit(X,y)

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x1_min, x1_max, 200),
        np.linspace(x2_min, x2_max, 200)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    predictions = model.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, predictions, alpha=0.25)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="black")
    plt.title(f"RBF SVM — C={c_value}")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()