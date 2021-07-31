import cv2
import numpy as np
from time import sleep

def max_to_write(mat, max):
    d = mat.copy()
    s = np.sum(d, axis=2, dtype='uint8')
    w = np.array(np.where(s<max, 0, 1), dtype='uint8')
    for i in range(mat.shape[2]):
        d[:,:,i] *= w
    # d = np.array(d, dtype='uint8')
    return d
def modify_contrast_and_brightness2(img, brightness=0 , contrast=100):
    # 上面做法的問題：有做到對比增強，白的的確更白了。
    # 但沒有實現「黑的更黑」的效果
    import math
    brightness = 0
    contrast = 100 # - 減少對比度/+ 增加對比度
    B = brightness / 255.0
    c = contrast / 255.0 
    k = math.tan((45 + 44 * c) / 180 * math.pi)
    img = (img - 127.5 * (1 - B)) * k + 127.5 * (1 + B)
    # 所有值必須介於 0~255 之間，超過255 = 255，小於 0 = 0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def sharpen(img, sigma=100):    
    # sigma = 5、15、25
    blur_img = cv2.GaussianBlur(img, (0, 0), sigma)
    usm = cv2.addWeighted(img, 1.5, blur_img, -0.5, 0)
    return usm

cap = cv2.VideoCapture(1)
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # hsv[:,:,0] = hsv[:,:,0] / 180 * 255
    # hsv[:,:,1] = 0
    # hsv[:,:,2] = 0
    # gray = hsv[:,:,0]
    gray[gray<50] = 255
    cv2.imshow('gray', gray)
    # sha = sharpen(gray, 100)
    # kernel = np.array([[-1, -1, -1], [-1, 7, -1], [-1, -1, -1]])
    # kernel = np.array( [[-0.125, -0.125, -0.125, -0.125, -0.125], 
    #                     [-0.125, 0.25, 0.25, 0.25, -0.125], 
    #                     [-0.125, 0.25, 1, 0.25, -0.125], 
    #                     [-0.125, 0.25, 0.25, 0.25, -0.125],
    #                     [-0.125, -0.125, -0.125, -0.125, -0.125]])
    # sha = cv2.filter2D(gray, -1, kernel=kernel, anchor=(-1, -1), delta=0, borderType=cv2.BORDER_DEFAULT)
    # sha = cv2.filter2D(sha, -1, kernel=kernel, anchor=(-1, -1), delta=0, borderType=cv2.BORDER_DEFAULT)
    # cv2.imshow('sha', sha)
    # kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    # kernel = np.array( [[0, 0, -1, 0, 0], 
    #                     [0, -1, -2, -1, 0], 
    #                     [-1, -2, 16, -2, -1], 
    #                     [0, -1, -2, -1, 0],
    #                     [0, 0, -1, 0, 0]])
    # result = cv2.filter2D(sha, -1, kernel=kernel, anchor=(-1, -1), delta=0, borderType=cv2.BORDER_DEFAULT)
    # cv2.imshow("Filter", result)
    # cv2.imshow('gray2', gray)
    # canny = cv2.Canny(hsv, 30, 150)
    # # cv2.imshow(f'canny', canny)
    # gray = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)
    # lower_green = np.array([0, 0, 0])
    # upper_green = np.array([50, 255, 255])
    # mask = cv2.inRange(hsv, lower_green, upper_green)
    # res = cv2.bitwise_and(frame, frame, mask=mask)
    # hsv_back = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray[gray<50]=255
    # frame = max_to_write(frame, 100)
    cv2.imshow('frame', frame)
    # cv2.imshow('hsv', hsv)
    # contrast = modify_contrast_and_brightness2(frame)
    # cv2.imshow('contrast', contrast)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

