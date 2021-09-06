import cv2
import numpy as np
from math import pow, sqrt

def get_distance(point_0, point_1):
    distance = pow((point_0[0] - point_1[0]), 2) + pow((point_0[1] - point_1[1]), 2)
    distance = sqrt(distance)
    return distance

def get_hog_descriptor(image):
    hog = cv2.HOGDescriptor()
    h, w = image.shape[:2]
    rate = 64 / w
    image = cv2.resize(image, (64, np.int(rate*h)))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bg = np.zeros((128, 64), dtype=np.uint8)
    bg[:,:] = 127
    h, w = gray.shape
    dy = (128 - h) // 2
    bg[dy:h+dy,:] = gray
    descriptors = hog.compute(bg, winStride=(8, 8), padding=(0, 0))
    return descriptors