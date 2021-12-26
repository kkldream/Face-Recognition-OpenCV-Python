import cv2
import numpy as np
import math
import utils
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from typing import Iterator
def fib(n: int) -> Iterator[int]:
	a, b = 0, 1
	while a < n:
		yield a
		a, b = b, a + b
		
def	test():
	print(fib(10))

def main():
	''' Create object '''
	cvFpsCalc = utils.CvFpsCalc(buffer_len=10)
	plt_1 = utils.PltOpenCV(100)
	''' Start Loop'''
	while True:
		display_fps = cvFpsCalc.get()
		# y = np.random.randint(0, 100)
		y = math.sin(time.time()*3)*100 + np.random.randint(-40, 40)
		# plt_img = plt_1(y)
		plt_1.append(y)
		b, a = signal.butter(30, 0.8, 'lowpass')   #配置濾波器 8 表示濾波器的階數
		filtedData = signal.filtfilt(b, a, plt_1.arr)
		plt_img = plt_1.draw()
		filted_img = plt_1.draw_arr(filtedData)
		cv2.putText(plt_img, "FPS:" + str(display_fps), (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
		cv2.imshow('plt_img', plt_img)
		cv2.imshow('filted_img', filted_img)
		if cv2.waitKey(1) == 27: # ESC
			break
	cv2.destroyAllWindows()


if __name__ == '__main__':
	# test()
	main()
