import cv2
import numpy as np
from time import sleep

def max_to_write(mat, max):
    d = mat.copy()
    s = np.sum(d, axis=2, dtype='uint8')
    w = np.array(np.where(s>max, 0, 1), dtype='uint8')
    for i in range(mat.shape[2]):
        d[:,:,i] *= w
    # d = np.array(d, dtype='uint8')
    return d

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = max_to_write(frame, 200)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()