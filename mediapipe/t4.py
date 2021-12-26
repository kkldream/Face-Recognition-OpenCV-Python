import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

arr = [0, 1, 5, 3, 2, 0, 2, 4, 6, 5, 3, 6, 6, 8, 2, 4, 5, 3, 2, 0, 4, 2, 3, 5, 0]
arr = [0]
for i in range(100):
    temp = arr[i] + random.randint(-3, 3)
    temp = max(0, min(temp, 10))
    arr.append(temp)
plt.subplot(121)
plt.plot(arr)


def cal_eye_times(arr):
    times = 0
    up = 0
    for i, _ in enumerate(arr[:-1]):
        a, b = arr[i], arr[i + 1]
        if b - a > 0:
            if up != 1: times += 1
            up = 1
        elif b - a < 0:
            up = -1
    return times

def draw_eye_times(arr):
    times = 0
    up = 0
    new_arr = arr.copy()
    pop_arr = []
    for i, _ in enumerate(arr[:-1]):
        a, b = arr[i], arr[i + 1]
        if b - a == 0:
            pop_arr.append(i)
            up == 0
        elif b - a > 0:
            if up != 1: times += 1
            else: pop_arr.append(i)
            up = 1
        elif b - a < 0:
            pass
            if up != -1: pass
            else: pop_arr.append(i)
            up = -1
    pop_arr.reverse()
    for i in pop_arr:
        new_arr.pop(i)
    return new_arr

print(f'times = {cal_eye_times(arr)}')
plt.subplot(122)
plt.plot(draw_eye_times(arr))
plt.show()

exit()

new_arr = arr.copy()
arr_pop_num = []
arr_up = []
up = 0
times = 0
for i, j in enumerate(new_arr[:-1]):
    a, b = j, new_arr[i + 1]
    if b - a == 0:
        print(f'{j} -> {new_arr[i + 1]}')
        arr_pop_num.append(i)
    elif b - a > 0:
        print(f'{j} -> {new_arr[i + 1]}, up')
        if up == 1:
            arr_pop_num.append(i)
        else:
            arr_up.append(1)
            times += 1
        up = 1
    elif b - a < 0:
        print(f'{j} -> {new_arr[i + 1]}, down')
        if up == -1:
            arr_pop_num.append(i)
        else:
            arr_up.append(0)
        up = -1
arr_pop_num.reverse()
for i in arr_pop_num:
    new_arr.pop(i)
plt.subplot(132)
plt.plot(new_arr)
plt.subplot(133)
plt.plot(arr_up)
print(f'times = {times}')
plt.show()
