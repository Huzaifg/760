import numpy as np

y = np.array([[2],[1]])
z = np.array([[1],[-1]])
X = np.array([[3,2],[-7,-5]])

print(y.T @ X @ z)


print(np.linalg.det(X))
print(np.linalg.inv(X))