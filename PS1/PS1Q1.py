import numpy as np
N = 15
roll = np.ceil(np.random.rand(N) * 6)
print(roll)
y = np.array([11, 22, 33, 44, 55, 66])
print(y)
z = y.reshape((3, 2))
print(z)
x = np.max(z)
print(x)
r, c = np.where(z == x)
print(r, c)
v = np.array([1, 4, 7, 1, 2, 6, 8, 1, 9])
x = np.sum(v == 1)
print(x)