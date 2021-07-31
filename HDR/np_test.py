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