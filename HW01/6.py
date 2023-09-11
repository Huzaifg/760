import numpy as np


# 3X3 matrix M
M = np.array([[5,0,0],[0,7,0],[0,0,3]])

# L2 norm of M
print(np.linalg.norm(M, ord=2))

# Forbenius norm of M
print(np.linalg.norm(M, ord='fro'))