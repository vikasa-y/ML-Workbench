import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# X = np.array([ [1],[2],[3],[4],[5],[6] ]) 
# (6 samples , 1 features so model.coef_[0])

#  it has (4 samples , 2 features) 
# so 1.model.coef_[0] /// 2.model.coef_[1]
X = np.array([[1, 50],[2, 60],[3, 70],[4, 80]])
y = np.array([35,42,50,58])

model = LinearRegression()

model.fit(X,y)

y_pred = model.predict(X)

print("w:",model.coef_)
print("Study Hours weight:", model.coef_[0])
print("Attendance weight:", model.coef_[1])
print("b:",model.intercept_)

mse = mean_squared_error(y, y_pred)
print("MSE:",mse)
print()

# ============================================
#       6 samples and 1 features examples
# ============================================

print("=="*30)
print("\n 2nd EXAMPLEso")
X = np.array([ [1],[2],[3],[4],[5],[6] ]) 
# (6 samples , 1 features so model.coef_[0])

y = np.array([35,42,50,58,64,72])

model = LinearRegression()

model.fit(X,y)

y_pred = model.predict(X)

print("w", model.coef_[0])
print("b:",model.intercept_)

mse = mean_squared_error(y, y_pred)
print("MSE:",mse)
