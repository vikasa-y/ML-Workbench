import numpy as np
actual = np.array([2,4,6,8,10])
predicted = np.array([4,7,10,13,16])

errors = actual - predicted

square_error = errors**2
print(errors)
print(square_error)

# =============================================
#   MSE = sum of sq error / number of samples
# =============================================

sum_square_error = square_error.sum()
MSE = (sum_square_error / 5)
print(MSE)