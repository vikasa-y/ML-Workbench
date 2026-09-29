import numpy as np

X = np.array([[1],[2],[3],[4],[5]])
y = np.array([2,4,6,8,10])

print(f"Shape of X : {X.shape}")
print(f"Shape of y : {y.shape}")

def predict(X,W,b):
    return W*X + b

predctions = predict(X,3,1)
print(predctions)