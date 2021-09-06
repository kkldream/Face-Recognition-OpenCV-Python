from lib.cv import cv2, CaptureInput
import mediapipe as mp
from math import atan, pi, pow, sqrt
from lib.header import *
from lib.face_mark import *
face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = CaptureInput(0, 640, 480, 30)

mouth_max = 1

while cap.isOpened():
	ret, frame = cap.read()
	results = face_mesh.process(frame)
	if results.multi_face_landmarks:
		for face_landmarks in results.multi_face_landmarks:
			# print(dir(face_landmarks))
			boxs = [0, 0, 0, 0]
			flm = FaceLandmark(face_landmarks, cap.shape)
			print(flm)
			left_eye_top = flm.get_coordinates('left_eye_top')
			left_eye_bottom = flm.get_coordinates('left_eye_bottom')
			right_eye_top = flm.get_coordinates('right_eye_top')
			right_eye_bottom = flm.get_coordinates('right_eye_bottom')
			mouth_top = flm.get_coordinates('mouth_top')
			mouth_bottom = flm.get_coordinates('mouth_bottom')
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

			
			mouth_distance = get_distance(mouth_top, mouth_bottom)
			mouth_max = max(mouth_distance, mouth_max)
			cv2.putText(frame, f'mouth = {int(mouth_distance / mouth_max * 100)}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)


			face_mid = (int((face_top[0] + face_bottom[0]) / 2), int((face_left[1] + face_right[1]) / 2))
			left_top = (int(face_left[0] - (face_mid[0] - face_top[0])), int(face_left[1] - (face_mid[1] - face_top[1])))
			right_top = (int(face_right[0] - (face_mid[0] - face_top[0])), int(face_right[1] - (face_mid[1] - face_top[1])))
			left_bottom = (int(face_left[0] - (face_mid[0] - face_bottom[0])), int(face_left[1] - (face_mid[1] - face_bottom[1])))
			right_bottom = (int(face_right[0] - (face_mid[0] - face_bottom[0])), int(face_right[1] - (face_mid[1] - face_bottom[1])))
			cv2.circle(frame, face_mid, 1, (255, 0, 0), -1)
			cv2.circle(frame, left_top, 1, (255, 0, 0), -1)
			cv2.circle(frame, right_top, 1, (255, 0, 0), -1)
			cv2.circle(frame, left_bottom, 1, (255, 0, 0), -1)
			cv2.circle(frame, right_bottom, 1, (255, 0, 0), -1)
			cv2.line(frame, left_top, right_top, (0, 255, 0), 1)
			cv2.line(frame, right_top, right_bottom, (0, 255, 0), 1)
			cv2.line(frame, right_bottom, left_bottom, (0, 255, 0), 1)
			cv2.line(frame, left_bottom, left_top, (0, 255, 0), 1)
			for i, data_point in enumerate(face_landmarks.landmark):
				pos = (int(cap.shape[0]*data_point.x), int(cap.shape[1]*data_point.y))
				if i == 0:
					boxs = [pos[0], pos[0], pos[1], pos[1]]
				else:
					boxs = [min(pos[0], boxs[0]), max(pos[0], boxs[1]), min(pos[1], boxs[2]), max(pos[1], boxs[3])]
				if i in [10, 13, 14, 127, 145, 152, 159, 356, 386, 374]:
					cv2.circle(frame, pos, 1, (0, 0, 255), -1)
				# 	cv2.putText(frame, str(i), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
				if data_point.z >= 0:
					cv2.circle(frame, pos, 2, (0, 0, int(255 * data_point.z * 50)), -1)
				else:
					cv2.circle(frame, pos, 2, (int(255 * data_point.z * -1 * 50), 0, 0), -1)
				# cv2.putText(frame, str(i), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
			# frame_copy = frame.copy()[boxs[2]-0:boxs[3]+0, boxs[0]-0:boxs[1]+0]
			# cv2.rectangle(frame, (boxs[0], boxs[2]), (boxs[1], boxs[3]), (0, 255, 0), 2)
			# print(boxs)
	cv2.imshow('MediaPipe FaceMesh', frame)
	# cv2.imshow('frame_copy', cv2.resize(frame_copy, (int(cap.shape[1] * 1.5), int(cap.shape[0] * 1.5))))
	if cv2.waitKey(50) & 0xFF == 27:
		break
cap.release()