import cv2
import numpy as np
import utils
import time
from face_mesh import FaceMesh
from mouth_detection import MouthDetection
import json
from keras.models import load_model
my_model = load_model('model.h5')

def main():
	level = 0
	percent_list = np.zeros((100, 1))
	''' CaptureInput '''
	cap = utils.CaptureInput(0, 640, 480, 30)
	# cap = utils.CaptureInput('blink_test.mp4', 960, 540)
	cap.setFlip = True
	''' Create object '''
	face_mesh = FaceMesh(1, 0.7, 0.7)
	mouth = MouthDetection()
	cvFpsCalc = utils.CvFpsCalc(buffer_len=10)
	''' Start Loop'''
	while True:
		display_fps = cvFpsCalc.get()
		ret, frame = cap.read()
		face_results = face_mesh(frame)
		for face_result in face_results:
			''' mouth detection '''
			percent = mouth(face_result, display_fps)
			mouth.draw_mouth_edge(frame, face_result)
			cv2.putText(frame, f'{int(percent)}%', face_result[81][:2],
						cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
			percent_list[:-1] = percent_list[1:]
			percent_list[-1] = percent
			train_mean = (1.809302, 29.645587)
			train_max = (5.000000, 85.310071)
			train_min = (0.00000, 2.04622)
			x = (percent_list - train_mean[1]) / (train_max[1] - train_min[1])
			predicted = my_model.predict(np.array([x]))
			level = max(0, min(5, int(predicted[0,0] * 10)))
			# print(level)

			# msg_dict = {
			# 	'level' : level,
			# 	'mouth' : percent
			# }
			# fileAppendTextLine(json.dumps(msg_dict))
		''' display '''
		cv2.putText(frame, "FPS:" + str(display_fps), (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
		cv2.putText(frame, "Level:" + str(level), (10, 60),
					cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
		cv2.imshow('frame', frame)
		key = cv2.waitKey(100)
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

if __name__ == '__main__':
	main()
