import cv2
import mediapipe as mp
import math
import utils
import numpy as np
from math import atan, pi, pow, sqrt
from lib.header import *
from lib.face_mark import *
from face_mesh import FaceMesh
from eye_blink import Eye
# face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = FaceMesh(1, 0.7, 0.7)
cap = utils.CaptureInput(0, 640, 480, 30)
eye = Eye()
left_eye_distance_list = np.zeros(cap.fps)
right_eye_distance_list = np.zeros(cap.fps)

def rotate_img(img, angle, center_pos):
    h, w, _ = img.shape
    # center = (w // 2, h // 2) # 找到圖片中心
    center = center_pos
    # 第一個參數旋轉中心，第二個參數旋轉角度(-順時針/+逆時針)，第三個參數縮放比例
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # 第三個參數變化後的圖片大小
    rotate_img = cv2.warpAffine(img, M, (w, h))
    return rotate_img

def rotate_xy(pos, center_pos, angle):
	x = pos[0]
	y = pos[1]
	cx = center_pos[0]
	cy = center_pos[1]
	radian = angle * pi / -180
	x_new = int((x - cx) * math.cos(radian) - (y - cy) * math.sin(radian) + cx)
	y_new = int((x - cx) * math.sin(radian) + (y - cy) * math.cos(radian) + cy)
	return x_new, y_new


while cap.isOpened():
	ret, frame = cap.read()
	# results = face_mesh.process(frame)
	face_results = face_mesh(frame)
	for face_result in face_results:
		roll, yaw, pitch = face_mesh.get_rotation(face_result)
		face_roi = face_mesh.get_align_roi(frame, face_result, roll)
		left_ear, right_ear = map(int, eye.get_eye_ear(face_result) * 100)
		eye.draw_eye_ear(frame, face_result)
		left_percent, right_percent = eye.info()

		cv2.putText(frame, f'{left_percent}%', face_result[29][:2], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255*left_percent/100), 2)
		cv2.putText(frame, f'{right_percent}%', face_result[258][:2], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255*right_percent/100), 2)
		break
	# if results.multi_face_landmarks:
	# 	for face_landmarks in results.multi_face_landmarks:
	# 		flm = FaceLandmark(face_landmarks, cap.shape)
	# 		left_eye_top = flm.get_coordinates('left_eye_top')
	# 		left_eye_bottom = flm.get_coordinates('left_eye_bottom')
	# 		right_eye_top = flm.get_coordinates('right_eye_top')
	# 		right_eye_bottom = flm.get_coordinates('right_eye_bottom')
	# 		mouth_top = flm.get_coordinates('mouth_top')
	# 		mouth_bottom = flm.get_coordinates('mouth_bottom')
	# 		face_top = flm.get_coordinates('face_top')
	# 		face_bottom = flm.get_coordinates('face_bottom')
	# 		face_left = flm.get_coordinates('face_left')
	# 		face_right = flm.get_coordinates('face_right')
	# 		left_eye_left_corner = flm.get_coordinates('left_eye_left_corner')
	# 		right_eye_right_corner = flm.get_coordinates('right_eye_right_corner')
	# 		between_mouth_nose = flm.get_coordinates('between_mouth_nose')
	# 		roll = int(atan((right_eye_right_corner[1] - left_eye_left_corner[1]) / (right_eye_right_corner[0] - left_eye_left_corner[0])) * 180 / pi)
	# 		yaw = int((right_eye_right_corner[2] - left_eye_left_corner[2]) * 400)
	# 		pitch = int((face_top[2] - between_mouth_nose[2]) * 400)

	# 		face_mid = (int((face_top[0] + face_bottom[0]) / 2), int((face_left[1] + face_right[1]) / 2))
	# 		left_top = (int(face_left[0] - (face_mid[0] - face_top[0])), int(face_left[1] - (face_mid[1] - face_top[1])))
	# 		right_top = (int(face_right[0] - (face_mid[0] - face_top[0])), int(face_right[1] - (face_mid[1] - face_top[1])))
	# 		left_bottom = (int(face_left[0] - (face_mid[0] - face_bottom[0])), int(face_left[1] - (face_mid[1] - face_bottom[1])))
	# 		right_bottom = (int(face_right[0] - (face_mid[0] - face_bottom[0])), int(face_right[1] - (face_mid[1] - face_bottom[1])))
	# 		roi_mid = int(max((right_bottom[0] + left_top[0]) / 2, (right_top[0] + left_bottom[0]) / 2)), int(max((left_bottom[1] + right_top[1]) / 2, right_bottom[1] + left_top[1]) / 2)
	# 		roi_wight = max(get_distance(right_bottom, left_top), get_distance(right_top, left_bottom))
	# 		roi_hidght = max(get_distance(left_bottom, left_top), get_distance(right_bottom, right_top))
	# 		roi_max_side = max(roi_wight, roi_hidght)

			# if results.multi_face_landmarks:
			# 	for face_landmarks in results.multi_face_landmarks:
			# 		for i, data_point in enumerate(face_landmarks.landmark):
			# 			pos = (int(cap.width * data_point.x), int(cap.height * data_point.y))
			# 			cv2.circle(frame, pos, 1, (0, 0, 255), -1)
			# cv2.circle(frame, left_eye_top[:2], 1, (255, 0, 0), -1)
			# cv2.circle(frame, left_eye_bottom[:2], 1, (255, 0, 0), -1)
			# cv2.circle(frame, right_eye_top[:2], 1, (255, 0, 0), -1)
			# cv2.circle(frame, right_eye_bottom[:2], 1, (255, 0, 0), -1)

		frame = rotate_img(frame, roll, roi_mid)
		boxs = (
			rotate_xy(left_top, roi_mid, roll),
			rotate_xy(right_top, roi_mid, roll),
			rotate_xy(right_bottom, roi_mid, roll),
			rotate_xy(left_bottom, roi_mid, roll)
		)
		face_roi = frame[
			boxs[0][1]:boxs[2][1],
			min(boxs[0][0], boxs[3][0]):max(boxs[1][0], boxs[2][0])
		]
		new_left_eye_top = rotate_xy(flm.get_coordinates('left_eye_top'), roi_mid, roll)
		new_left_eye_bottom = rotate_xy(flm.get_coordinates('left_eye_bottom'), roi_mid, roll)
		new_left_eye_left_corner = rotate_xy(flm.get_coordinates('left_eye_left_corner'), roi_mid, roll)
		new_left_eye_right_corner = rotate_xy(flm.get_coordinates('left_eye_right_corner'), roi_mid, roll)
		new_right_eye_top = rotate_xy(flm.get_coordinates('right_eye_top'), roi_mid, roll)
		new_right_eye_bottom = rotate_xy(flm.get_coordinates('right_eye_bottom'), roi_mid, roll)
		new_right_eye_left_corner = rotate_xy(flm.get_coordinates('right_eye_left_corner'), roi_mid, roll)
		new_right_eye_right_corner = rotate_xy(flm.get_coordinates('right_eye_right_corner'), roi_mid, roll)
		# left_eye_mid = int((new_left_eye_left_corner[0] + new_left_eye_right_corner[0]) / 2), int((new_left_eye_top[1] + new_left_eye_bottom[1]) / 2)
		left_eye_mid = int((new_left_eye_top[0] + new_left_eye_bottom[0]) / 2), int((new_left_eye_top[1] + new_left_eye_bottom[1]) / 2)
		# right_eye_mid = int((new_right_eye_left_corner[0] + new_right_eye_right_corner[0]) / 2), int((new_right_eye_top[1] + new_right_eye_bottom[1]) / 2)
		right_eye_mid = int((new_right_eye_top[0] + new_right_eye_bottom[0]) / 2), int((new_right_eye_top[1] + new_right_eye_bottom[1]) / 2)
		cv2.circle(frame, left_eye_mid, 1, (0, 0, 255), -1)
		cv2.circle(frame, right_eye_mid, 1, (0, 0, 255), -1)

		left_eye_distance = get_distance(left_eye_top, left_eye_bottom)
		right_eye_distance = get_distance(right_eye_top, right_eye_bottom)

		left_eye_distance_list[1:] = left_eye_distance_list[:-1]
		left_eye_distance_list[0] = left_eye_distance
		left_eye_distance_avg = np.average(left_eye_distance_list)
		if left_eye_distance_list[0] / left_eye_distance_avg < 0.85:
			cv2.circle(frame, left_eye_mid, 5, (0, 0, 255), 1)
			# cv2.putText(frame, f'left_eye_blink', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
			
		right_eye_distance_list[1:] = right_eye_distance_list[:-1]
		right_eye_distance_list[0] = right_eye_distance
		right_eye_distance_avg = np.average(right_eye_distance_list)
		if right_eye_distance_list[0] / right_eye_distance_avg < 0.85:
			cv2.circle(frame, right_eye_mid, 5, (0, 0, 255), 1)
			# cv2.putText(frame, f'right_eye_blink', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

		left_eye_roi = frame.copy()[new_left_eye_top[1] - 5:new_left_eye_bottom[1] + 5, new_left_eye_left_corner[0]:new_left_eye_right_corner[0]]
		right_eye_roi = frame.copy()[new_right_eye_top[1] - 5:new_right_eye_bottom[1] + 5, new_right_eye_left_corner[0]:new_right_eye_right_corner[0]]

		
		cv2.putText(frame, f'roll = {roll}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
		cv2.putText(frame, f'yaw = {yaw}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
		cv2.putText(frame, f'pitch = {pitch}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
		cv2.imshow('left_eye_roi', cv2.resize(left_eye_roi, (int(left_eye_roi.shape[1] * 10), int(left_eye_roi.shape[0] * 10))))
		# cv2.imshow('right_eye_roi', cv2.resize(right_eye_roi, (int(right_eye_roi.shape[1] * 10), int(right_eye_roi.shape[0] * 10))))
		# canny = cv2.Canny(left_eye_roi, 30, 150)
		# cv2.imshow('canny', cv2.resize(canny, (int(canny.shape[1] * 10), int(canny.shape[0] * 10))))

		cv2.imshow('face_roi', cv2.resize(face_roi, (int(face_roi.shape[1] * 3), int(face_roi.shape[0] * 3))))
		# hog = get_hog_descriptor(face_roi)
		# print(hog.shape)

	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == 27:
		break
cap.release()