import numpy as np

X = np.array([1,2,3,4,5])

w = 1
b = -3

z = w*X +b

probability = 1 / (1+np.exp(-z))

print("Probability:",probability)

y_pred = (probability >= 0.5).astype(int)
print("Prediction:",y_pred)

# it is called as Logistic Regression,
#  but it's mainly used for classification.