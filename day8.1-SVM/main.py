import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score


X = np.array([[1],[2],[3],[4],[5],[6],[7],[8]])
y = np.array([0,0,0,0,1,1,1,1])

X_train , X_test  , y_train , y_test = train_test_split(X,y,test_size=0.25,random_state=42)

model = make_pipeline(StandardScaler(),SVC(kernel="linear" , C=100))

model.fit(X_train , y_train)

y_pred = model.predict(X_test)

print("Prediction :",y_pred)
print("Actal :",y_test)
print("Accuracy :", accuracy_score(y_test , y_pred))

for c_value in [0 , 0.01 , 100]:
    Model = SVC(kernel="linear" , C=c_value)
    model.fit(X,y)
    print("C:", c_value)
    print("New Prediction :", model.predict([[3.5], [4.5], [5.5]]))