import numpy as np

import matplotlib

a = np.array([[-0.3,0.2,0.04],
             [0.12,-0.11,-0.7],
             [0.03,-0.2,0.33]])

b = np.array([12.3,-1.23,23.23]).transpose()
bt = np.array([[12.3],[-1.23],[23.23]])
t = np.array([0.2,0.3,0.5])

print(bt)
print(b)
print(b.shape, bt.shape)
print(a.dot(b) + t)
print(a@b +t )
