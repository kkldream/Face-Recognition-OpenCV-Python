from lib.cv import cv2, CaptureInput
import mediapipe as mp
from lib.display_fps import DisplayFPS

face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = CaptureInput(0, 640, 480, 30)
fps = DisplayFPS()

while cap.isOpened():
	ret, frame = cap.read()
	frame = cv2.flip(frame, 1)
	results = face_mesh.process(frame)
	if results.multi_face_landmarks:
		for face_landmarks in results.multi_face_landmarks:
			boxs = [0, 0, 0, 0]
			for i, data_point in enumerate(face_landmarks.landmark):
				pos = (int(cap.shape[0]*data_point.x), int(cap.shape[1]*data_point.y))
				if i == 0:
					boxs = [pos[0], pos[0], pos[1], pos[1]]
				else:
					boxs = [min(pos[0], boxs[0]), max(pos[0], boxs[1]), min(pos[1], boxs[2]), max(pos[1], boxs[3])]
				if data_point.z >= 0:
					cv2.circle(frame, pos, 2, (0, 0, int(255 * data_point.z * 50)), -1)
				else:
					cv2.circle(frame, pos, 2, (int(255 * data_point.z * -1 * 50), 0, 0), -1)
			print(boxs)
	fps.count()
	fps.print(frame)
	cv2.imshow('MediaPipe FaceMesh', frame)
	if cv2.waitKey(50) & 0xFF == 27:
		break
cap.release()