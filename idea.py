import numpy as np

x = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1,2,3,4,5,6,7,8,9,10]])
y = (x[1,3], x[1,4])
print(y)
# print(len(x) * x.shape[1])