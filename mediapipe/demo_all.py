import cv2
import utils
import numpy as np
from face_mesh import FaceMesh
from eye_detection import EyeDetection
from face_classification import FaceClassification
from mouth_detection import MouthDetection
import time

def main():
	''' CaptureInput '''
	cap = utils.CaptureInput(0, 640, 480, 15)
	cap.setFlip = True
	''' Create object '''
	face_mesh = FaceMesh(1, 0.7, 0.7)
	eye = EyeDetection()
	cvFpsCalc = utils.CvFpsCalc(buffer_len=10)
	face_classification = FaceClassification()
	mouth = MouthDetection()
	plt_1 = utils.PltOpenCV(100)
	head_wait_time = 0
	head_last_pos = (0, 0)
	eye_kalman = utils.KalmanFilter(1, 0.01)
	mouth_kalman = utils.KalmanFilter(1, 0.01)
	while cap.isOpened():
		display_fps = cvFpsCalc.get()
		ret, frame = cap.read()
		face_results = face_mesh(frame)
		for face_result in face_results:
			''' get face bbox'''
			face_bbox = face_mesh.calc_face_bbox(face_result)
			face_mid = face_mesh.calc_face_mid(face_result)
			''' get rotation '''
			roll, yaw, pitch = face_mesh.get_rotation(face_result)
			''' get align roi '''
			face_roi = face_mesh.get_align_roi(frame, face_result, roll)
			face_roi = face_classification.get_resize(face_roi)
			''' face classification '''
			head_pos = face_mesh.calc_face_mid(face_result)
			move_rate = utils.get_distance(head_pos, head_last_pos) / (face_bbox[1][0] - face_bbox[0][0])
			if time.time() - head_wait_time > 10 or move_rate > 0.5:
				head_wait_time = time.time()
				head_last_pos = head_pos
				name, credibility = face_classification(face_roi, roll)
			elif name == 'Unknown':
				head_wait_time *= 0.5
			''' face rotate bbox '''
			all_face_bbox = [
				face_bbox[0],
				(face_bbox[0][0], face_bbox[1][1]),
				face_bbox[1],
				(face_bbox[1][0], face_bbox[0][1])
			]
			rotate_face_bbox = [utils.rotate_xy(all_face_bbox[i], face_mid, roll * -1) for i in range(4)]
			for i in range(-1, 3):
				cv2.line(frame, rotate_face_bbox[i], rotate_face_bbox[i+1], (255, 0, 0))
			''' mouth detection '''
			mouth_percent = mouth(face_result, display_fps)
			mouth.draw_mouth_edge(frame, face_result)
			''' detection eye '''
			head_pos = face_mesh.calc_face_mid(face_result)
			move_rate = utils.get_distance(head_pos, head_last_pos) / (face_bbox[1][0] - face_bbox[0][0])
			head_last_pos = head_pos
			if move_rate < 0.05:
				eye.draw_eye_edge(frame, face_result)
				left_eye_percent, right_eye_percent = eye(face_result, display_fps)
				cv2.putText(frame, f'{left_eye_percent}%', face_result[29][:2],
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255*left_eye_percent/100), 2)
				cv2.putText(frame, f'{right_eye_percent}%', face_result[258][:2],
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255*right_eye_percent/100), 2)
			''' display '''
			cv2.imshow('face_roi', face_roi)
			cv2.putText(frame, f'{int(mouth_percent)}%', face_result[140][:2],
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255*mouth_percent/100), 2)
			cv2.putText(frame, f'{name}:{int(credibility)}%', (10, 60),
						cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
			cv2.putText(frame, f'roll:{int(roll)}', (10, 90),
						cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
			cv2.putText(frame, f'yaw:{int(yaw)}', (10, 120),
						cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
			cv2.putText(frame, f'pitch:{int(pitch)}', (10, 150),
						cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
			total = abs(roll) + abs(yaw) + abs(pitch)
			cv2.putText(frame, f'total:{int(total)}', (10, 180),
						cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255*min(1, total/50)), 2)
			''' kalman '''
			# plt_img = plt_1(mouth_prediction, 0, 101)
			# cv2.imshow('plt_img', plt_img)
			# if abs(pitch) < 20:
			# 	if yaw >= 0:
			# 		eye_prediction = left_eye_percent
			# 	else:
			# 		eye_prediction = right_eye_percent
			# 	print(plt_1.arr)
			# 	print(cal_eye_times(plt_1.arr))
			# 	plt_img = plt_1(eye_prediction, 0, 101)
			# 	cv2.imshow('plt_img', plt_img)
		cv2.putText(frame, "FPS:" + str(display_fps), (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
		cv2.imshow('frame', frame)
		if cv2.waitKey(1) & 0xFF == 27:
			break
	cap.release()
	cv2.destroyAllWindows()

def cal_eye_times(arr):
	r = 2
	times = 0
	up = 0
	for i, _ in enumerate(arr[:-1]):
		a, b = arr[i], arr[i + 1]
		if b - a > r:
			if up != 1: times += 1
			up = 1
		elif b - a < r:
			up = -1
	return times
	
if __name__ == '__main__':
	main()
