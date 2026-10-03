import numpy as np

X = np.array([1,2,3,7,8,9])
y = np.array([0,0,0,1,1,1])

new_point = 6
k = 3

distance = np.abs(X - new_point)

indices = np.argsort(distance)

near_indices = indices[:k]

nearest_labels = y[near_indices]

prediction = np.bincount(nearest_labels).argmax()

print("Distance :", distance)
print("Nearest indices :", near_indices)
print("Nearest labels :",nearest_labels)
print("Prediction :",prediction)