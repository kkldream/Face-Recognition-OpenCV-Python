import cv2
import numpy as np
import math
import utils

buf1 = np.array([
    [0, 1],
    [2, 5],
    [1, 1]
])
buf2 = np.array([
    [5, 1],
    [2, 1],
    [10, 8]
])
print(max(buf1, buf2))