import cv2
import mediapipe as mp
from display_fps import DisplayFPS
from math import atan, pi, pow, sqrt

face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('test.mp4')
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 12)
cap_shape = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
cap_fps = int(cap.get(cv2.CAP_PROP_FPS))
print(cap_shape, cap_fps)
fps = DisplayFPS()

def get_landmark_pos(num, shape):
	landmark = face_landmarks.landmark[num]
	pos = (int(shape[0] * landmark.x), int(shape[1] * landmark.y), landmark.z)
	return pos

def get_distance(point_0, point_1):
    distance = pow((point_0[0] - point_1[0]),2) + pow((point_0[1] - point_1[1]),2)
    distance = sqrt(distance)
    return distance


mouth_max = 0

while cap.isOpened():
	ret, frame = cap.read()
	frame = cv2.flip(frame, 1)
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	results = face_mesh.process(frame)
	frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
	if results.multi_face_landmarks:
		for face_landmarks in results.multi_face_landmarks:
			# print(dir(face_landmarks))
			boxs = [0, 0, 0, 0]
			left_eye_top = get_landmark_pos(159, cap_shape)
			left_eye_bottom = get_landmark_pos(145, cap_shape)
			right_eye_top = get_landmark_pos(386, cap_shape)
			right_eye_bottom = get_landmark_pos(145, cap_shape)
			mouth_top = get_landmark_pos(13, cap_shape)
			mouth_bottom = get_landmark_pos(14, cap_shape)
			face_top = get_landmark_pos(10, cap_shape)
			face_bottom = get_landmark_pos(152, cap_shape)
			face_left = get_landmark_pos(127, cap_shape)
			face_right = get_landmark_pos(356, cap_shape)
			left_eye_edge = get_landmark_pos(226, cap_shape)
			right_eye_edge = get_landmark_pos(446, cap_shape)
			under_nose = get_landmark_pos(164, cap_shape)
			
			roll = int(atan((right_eye_edge[1] - left_eye_edge[1]) / (right_eye_edge[0] - left_eye_edge[0])) * 180 / pi)
			yaw = int((right_eye_edge[2] - left_eye_edge[2]) * 400)
			pitch = int((face_top[2] - under_nose[2]) * 400)
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
				pos = (int(cap_shape[0]*data_point.x), int(cap_shape[1]*data_point.y))
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
	fps.count()
	fps.print(frame)
	cv2.imshow('MediaPipe FaceMesh', frame)
	# cv2.imshow('frame_copy', cv2.resize(frame_copy, (int(cap_shape[1] * 1.5), int(cap_shape[0] * 1.5))))
	if cv2.waitKey(50) & 0xFF == 27:
		break
cap.release()