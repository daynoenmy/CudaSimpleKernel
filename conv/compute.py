import numpy as np

x = np.array([1,2,3,4,5,6,7])
y = np.array([1,2,3,4,5])


z = np.convolve(x,y,'same')

print(z)