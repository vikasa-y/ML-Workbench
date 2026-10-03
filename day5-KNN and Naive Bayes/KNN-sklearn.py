from sklearn.neighbors import KNeighborsClassifier
import numpy as np

X = np.array([[1],[2],[3],[7],[8],[9] ])
y = np.array([0,0,0,1,1,1])

model = KNeighborsClassifier(n_neighbors=3)

model.fit(X,y)

new_point = np.array([[6]])

prediction = model.predict(new_point)
print(prediction)