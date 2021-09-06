from lib.cv import cv2, CaptureInput
import mediapipe as mp
import math
from math import atan, pi, pow, sqrt
from lib.header import *
from lib.face_mark import *
face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = CaptureInput(0, 640, 480, 30)

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
	results = face_mesh.process(frame)
	if results.multi_face_landmarks:
		for face_landmarks in results.multi_face_landmarks:
			flm = FaceLandmark(face_landmarks, cap.shape)
			face_top = flm.get_coordinates('face_top')
			face_bottom = flm.get_coordinates('face_bottom')
			face_left = flm.get_coordinates('face_left')
			face_right = flm.get_coordinates('face_right')
			left_eye_corner = flm.get_coordinates('left_eye_corner')
			right_eye_corner = flm.get_coordinates('right_eye_corner')
			between_mouth_nose = flm.get_coordinates('between_mouth_nose')
			roll = int(atan((right_eye_corner[1] - left_eye_corner[1]) / (right_eye_corner[0] - left_eye_corner[0])) * 180 / pi)
			yaw = int((right_eye_corner[2] - left_eye_corner[2]) * 400)
			pitch = int((face_top[2] - between_mouth_nose[2]) * 400)
			cv2.putText(frame, f'roll = {roll}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
			cv2.putText(frame, f'yaw = {yaw}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
			cv2.putText(frame, f'pitch = {pitch}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

			face_mid = (int((face_top[0] + face_bottom[0]) / 2), int((face_left[1] + face_right[1]) / 2))
			left_top = (int(face_left[0] - (face_mid[0] - face_top[0])), int(face_left[1] - (face_mid[1] - face_top[1])))
			right_top = (int(face_right[0] - (face_mid[0] - face_top[0])), int(face_right[1] - (face_mid[1] - face_top[1])))
			left_bottom = (int(face_left[0] - (face_mid[0] - face_bottom[0])), int(face_left[1] - (face_mid[1] - face_bottom[1])))
			right_bottom = (int(face_right[0] - (face_mid[0] - face_bottom[0])), int(face_right[1] - (face_mid[1] - face_bottom[1])))
			roi_mid = int(max((right_bottom[0] + left_top[0]) / 2, (right_top[0] + left_bottom[0]) / 2)), int(max((left_bottom[1] + right_top[1]) / 2, right_bottom[1] + left_top[1]) / 2)
			roi_wight = max(get_distance(right_bottom, left_top), get_distance(right_top, left_bottom))
			roi_hidght = max(get_distance(left_bottom, left_top), get_distance(right_bottom, right_top))
			roi_max_side = max(roi_wight, roi_hidght)

			frame = rotate_img(frame, roll, roi_mid)
			boxs = (
				rotate_xy(left_top, roi_mid, roll),
				rotate_xy(right_top, roi_mid, roll),
				rotate_xy(right_bottom, roi_mid, roll),
				rotate_xy(left_bottom, roi_mid, roll)
			)
			face_roi = frame.copy()[
				boxs[0][1]:boxs[2][1],
				min(boxs[0][0], boxs[3][0]):max(boxs[1][0], boxs[2][0])
			]
	cv2.imshow('face_roi', face_roi)
	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == 27:
		break
cap.release()