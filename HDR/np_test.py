import numpy as np

def max_to_write(mat, max):
    d = mat.copy()
    s = np.sum(d, axis=2)
    w = np.where(s>max, 0, 1)
    for i in range(mat.shape[2]):
        d[:,:,i] *= w
    return d

arr = np.array([[[1,2,3],[2,3,4]],[[1,2,3],[2,3,4]],[[1,2,3],[2,3,4]],[[1,2,3],[2,3,4]]])
dst = max_to_write(arr, 8)
print(arr)
print(arr.shape)
print(dst)
print(dst.shape)
# s = np.sum(arr, axis=2)
# # s = np.where(sum(arr[:,:])>10,0,arr)
# print(s)
# # print(s.shape)
# wh = np.where(s>8,0,1)
# print(wh)
# # print(arr[:,:,0] * wh)
# arr[:,:,0] *= wh
# arr[:,:,1] *= wh
# arr[:,:,2] *= wh
# # # arr[arr>3] = 0
# print(arr)


# import cv2
# image = cv2.imread('img_15.jpg')
# print(image)
# cv2.imshow('image', image)
# where = np.where(np.sum(image,axis=2)>500,0,image)
# print(where)
# cv2.imshow('where', where)
# cv2.waitKey(0)