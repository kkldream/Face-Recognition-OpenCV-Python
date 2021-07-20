import cv2
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

exit()

import cv2
import numpy as np
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
original_size = cap.read()[1].shape
cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
convert_size = cap.read()[1].shape
print(f'original: {original_size}')
print(f'convert: {convert_size}')
while(True):
    ret, frame = cap.read()
    # new_frame = []
    for i in range(len(frame[0])):
        frame[0][i] = frame[0][i - i % 4]
    # data = np.resize(new_frame,(480,640,1),dtype=np.uint8)
    # print(frame.shape)
    # cv2.imshow('frame', data)
    # print(new_frame.shape)
    # img = frame.reshape((1920, 640, 1))
    # cv2.imshow('frame', frame)
    img = frame.reshape((480, 640, 4))
    cv2.imshow('frame', img)
    # cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()