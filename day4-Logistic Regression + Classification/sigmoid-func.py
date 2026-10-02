# Classification
"""
In Regression: 
Predict a continuous number.

In Classification:
Predict a category/class.
"""

import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

print(sigmoid(-5))
print(sigmoid(0))
print(sigmoid(5))