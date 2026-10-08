import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X = np.array([[1],[2],[3],[4],[5],[6],[7],[8]])

y = np.array([0,0,0,0,1,1,1,1])

X_train, X_test , y_train, y_test = train_test_split(
    X,y,
    test_size=0.2,
    random_state=42
)

model_1 = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

model_2 = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,   # maximum deapth of each tree
    random_state=42
)

# training a models

model_1.fit(X_train , y_train)
model_2.fit(X_train , y_train)

#  predictions 
y_predict_of_model_1 = model_1.predict(X_test)
y_predict_of_model_2 = model_2.predict(X_test)


print("\n------Decision Tree------\n")
print("predictions :",y_predict_of_model_1)
print("Actual :",y_test)

new_student = np.array([[4.5]])
print("Prediction:", model_1.predict(new_student))
print("Probability:", model_1.predict_proba(new_student))

print("Accuarcy:", accuracy_score(y_test , y_predict_of_model_1))

print()
dt_train_pred = model_1.predict(X_train)
dt_test_pred = model_1.predict(X_test)

print("Decision Tree Training Accuracy:",
      accuracy_score(y_train, dt_train_pred))

print("Decision Tree Testing Accuracy:",
      accuracy_score(y_test, dt_test_pred))

print("="*80)

print("\n------Random Forest------\n")
print("predictions :",y_predict_of_model_2)
print("Actual :",y_test)


print("Prediction of model_2 :", model_2.predict(new_student))
print("Probability of model_2 :", model_2.predict_proba(new_student))

print("Accuarcy of Model_2 :", accuracy_score(y_test , y_predict_of_model_2))

rf_train_pred = model_2.predict(X_train)
rf_test_pred = model_2.predict(X_test)

print("Random Forest Training Accuracy:",
      accuracy_score(y_train, rf_train_pred))

print("Random Forest Testing Accuracy:",
      accuracy_score(y_test, rf_test_pred))
print("\n")

