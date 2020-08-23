import numpy as np

A = np.array([[1, 0, 1],[2, -1, 1],[-3, 2, -2]])
b = np.array([[-2], [1], [-1]])
Ainv = np.linalg.inv(A)
x = np.matmul(Ainv, b)
print(x)