from sklearn.naive_bayes import GaussianNB
import numpy as np

X = np.array([
    [1],
    [2],
    [3],
    [7],
    [8],
    [9]
])

y = np.array([0,0,0,1,1,1])

model = GaussianNB()

model.fit(X , y)

new_point = np.array([[5]])

prediction = model.predict(new_point)
print(prediction)

probability = model.predict_proba(new_point)
print(probability)