import matplotlib.pyplot as plt
import numpy as np

arr = [1,2,3,3,4,5,6]
print(arr * 2)
x = np.linspace(0, len(arr) - 1, len(arr))
y = arr
# plt.hist(arr, 256, [0, 256])
print(x)
print(y)
plt.hist(x, y)
plt.plot(y)
plt.show()

