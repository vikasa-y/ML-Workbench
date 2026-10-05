from sklearn.tree import DecisionTreeClassifier
import numpy as np

X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [8]
])

y = np.array([0,0,0,1,1,1,1])

model = DecisionTreeClassifier()
model.fit(X , y)

new_student = np.array([[7]])
new_student_probability = model.predict_proba([[7]])
# new_student = np.array([7])
# This become an error Scikit-learn expects:(samples, features)

prediction = model.predict(new_student)
print(prediction)
print(f"Probability : ",new_student_probability)