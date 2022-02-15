import cv2
import numpy as np
import utils
import time
from face_mesh import FaceMesh
import json

def main():
	''' CaptureInput '''
	cap = utils.CaptureInput(0, 640, 480, 30)
	# cap = utils.CaptureInput('blink_test.mp4', 960, 540)
	cap.setFlip = True
	''' Create object '''
	face_mesh = FaceMesh(1, 0.7, 0.7)
	cvFpsCalc = utils.CvFpsCalc(buffer_len=10)
	''' Start Loop'''
	while True:
		display_fps = cvFpsCalc.get()
		ret, frame = cap.read()
		face_results = face_mesh(frame)
		for face_result in face_results:
			''' mouth detection '''
			mouth_ear = calc_mouth_ear(face_result)
			frame = draw_mouth_edge(frame.copy(), face_result)
			cv2.putText(frame, f'{int(mouth_ear*100)}%', face_result[81][:2],
						cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
		''' display '''
		cv2.putText(frame, "FPS:" + str(display_fps), (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
		cv2.imshow('frame', frame)
		key = cv2.waitKey(1)
		if key == 27: # ESC
			break
		for i in range(10):
			if key == ord(str(i)):
				level = i
				print(f'level = {level}')
	cap.release()
	cv2.destroyAllWindows()

def fileAppendTextLine(msg):
    with open('output2.txt', 'a', encoding='utf-8') as file:
        file.write(f'{msg}\n')

def calc_mouth_ear(landmarks):
	p1 = landmarks[61]
	p2 = landmarks[81]
	p3 = landmarks[311]
	p4 = landmarks[291]
	p5 = landmarks[402]
	p6 = landmarks[178]
	ear = (utils.get_distance(p2, p6) + utils.get_distance(p3, p5))\
			/ (2 * utils.get_distance(p1, p4))
	return ear

def draw_mouth_edge(image, landmarks):
	pos = (61, 81, 311, 291, 402, 178)
	for p in pos:
		cv2.circle(image, landmarks[p][:2], 2, (0, 0, 255), -1)
	return image

def map(var, in_min, in_max, out_min, out_max):
    var = float(var)
    if var >= in_max:
        return out_max
    if var <= in_min:
        return out_min
    return (var - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

if __name__ == '__main__':
	main()
