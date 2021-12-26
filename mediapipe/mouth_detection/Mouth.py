import cv2
import numpy as np
import math
import utils

class MouthDetection:
	max_open = 0
	avg = 0

	def __init__(self, buf=100) -> None:
		self.buf = np.zeros(buf)
		
	def __call__(self, landmarks, fps):
		self._calc_eye_ear(landmarks)
		# rate = (0.999 ** 30) ** (1 / fps)
		rate = 1
		self.max_open = max(self.max_open * rate, self.buf[0])
		self.percent = self.buf[0] / self.max_open
		# self._calc_detection(fps)
		return self.percent * 100

	def _calc_eye_ear(self, landmarks):
		p1 = landmarks[61]
		p2 = landmarks[81]
		p3 = landmarks[311]
		p4 = landmarks[291]
		p5 = landmarks[402]
		p6 = landmarks[178]
		ear = (utils.get_distance(p2, p6) + utils.get_distance(p3, p5))\
				/ (2 * utils.get_distance(p1, p4))
		self.buf[1:-1] = self.buf[:-2]
		self.buf[0] = ear

	def draw_mouth_edge(self, image, landmarks):
		pos = [
			landmarks[61],
			landmarks[81],
			landmarks[311],
			landmarks[291],
			landmarks[402],
			landmarks[178]
		]
		for p in pos:
			cv2.circle(image, p[:2], 1, (0, 0, 255), -1)
