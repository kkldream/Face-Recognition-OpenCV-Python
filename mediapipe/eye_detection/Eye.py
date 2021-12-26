import cv2
import numpy as np
import math
import utils

class EyeDetection:
	max_open = [0, 0]
	max_dis = [0.001, 0.001]
	avg = [0, 0]

	def __init__(self, buf=100) -> None:
		self.buf = np.zeros((buf, 2))
		
	def __call__(self, landmarks, fps):
		self._calc_eye_ear(landmarks)
		self._calc_detection(fps)
		return self.percent

	def _calc_eye_ear(self, landmarks):
		p1 = landmarks[33]
		p2 = landmarks[160]
		p3 = landmarks[158]
		p4 = landmarks[133]
		p5 = landmarks[153]
		p6 = landmarks[144]
		left_ear = (utils.get_distance(p2, p6) + utils.get_distance(p3, p5))\
				/ (2 * utils.get_distance(p1, p4))
		p1 = landmarks[263]
		p2 = landmarks[387]
		p3 = landmarks[385]
		p4 = landmarks[362]
		p5 = landmarks[380]
		p6 = landmarks[373]
		right_ear = (utils.get_distance(p2, p6) + utils.get_distance(p3, p5))\
				/ (2 * utils.get_distance(p1, p4))
		self.buf[1:] = self.buf[:-1]
		self.buf[0] = left_ear, right_ear

	def _calc_detection(self, fps):
		rate = (0.999 ** 30) ** (1 / fps)
		# print(rate)
		self.avg = np.average(self.buf, 0)
		self.max_open = max(self.max_open[0] * rate, self.avg[0]),\
						max(self.max_open[1] * rate, self.avg[1])
		self.max_dis = max(self.max_dis[0] * rate, self.max_open[0] - self.buf[0][0]),\
					   max(self.max_dis[1] * rate, self.max_open[1] - self.buf[0][1])
		self.percent = int(max(0, (self.max_open[0] - self.buf[0][0]) / self.max_dis[0]) * 100),\
					   int(max(0, (self.max_open[1] - self.buf[0][1]) / self.max_dis[1]) * 100)

	def draw_eye_edge(self, image, landmarks):
		pos = [
			landmarks[33],
			landmarks[160],
			landmarks[158],
			landmarks[133],
			landmarks[153],
			landmarks[144],
			landmarks[263],
			landmarks[387],
			landmarks[385],
			landmarks[362],
			landmarks[380],
			landmarks[373]
		]
		for p in pos:
			cv2.circle(image, p[:2], 1, (0, 0, 255), -1)

	def info(self):
		self.calc_ear_avg()
		left_percent = int(max(0, (self.max_open[0] - self.buf[0][0]) / self.max_dis[0]) * 100)
		right_percent = int(max(0, (self.max_open[1] - self.buf[0][1]) / self.max_dis[1]) * 100)
		print(
			f'percent = {left_percent}%, {right_percent}%\n'
			f'{int(self.buf[0][0] * 100), int(self.buf[0][1] * 100)}\n'
			f'avg = {int(self.avg[0] * 100), int(self.avg[1] * 100)}\n'
			f'max_open = {int(self.max_open[0] * 100), int(self.max_open[1] * 100)}\n'
			f'max_dis = {int(self.max_dis[0] * 100), int(self.max_dis[1] * 100)}\n'
			# f'max_dis = {self.max_open[0] - self.avg[0]}\n'
			f'----------------------------------------'
		)
		return left_percent, right_percent
