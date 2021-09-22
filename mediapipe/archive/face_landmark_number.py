from lib.cv import cv2, CaptureInput
import mediapipe as mp
from lib.display_fps import DisplayFPS

face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = CaptureInput(0, 640, 480, 30)
fps = DisplayFPS()

def get_landmark_pos(num, shape):
	landmark = face_landmarks.landmark[num]
	pos = (int(shape[0] * landmark.x), int(shape[1] * landmark.y))
	return pos

while cap.isOpened():
	ret, frame = cap.read()
	frame = cv2.flip(frame, 1)
	results = face_mesh.process(frame)
	if results.multi_face_landmarks:
		for face_landmarks in results.multi_face_landmarks:
			boxs = [0, 0, 0, 0]
			for i, data_point in enumerate(face_landmarks.landmark):
				pos = (int(cap.width * data_point.x), int(cap.height * data_point.y))
				if i == 0:
					boxs = [pos[0], pos[0], pos[1], pos[1]]
				else:
					boxs = [min(pos[0], boxs[0]), max(pos[0], boxs[1]), min(pos[1], boxs[2]), max(pos[1], boxs[3])]
				cv2.circle(frame, pos, 1, (0, 0, 255), -1)
				# cv2.putText(frame, str(i), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
			# frame_copy = frame.copy()[boxs[2]-0:boxs[3]+0, boxs[0]-0:boxs[1]+0]
			cv2.rectangle(frame, (boxs[0], boxs[2]), (boxs[1], boxs[3]), (0, 255, 0), 2)
	fps.count()
	fps.print(frame)
	cv2.imshow('MediaPipe FaceMesh', frame)
	# cv2.imshow('frame_copy', cv2.resize(frame_copy, (int(cap_shape[1] * 1.5), int(cap_shape[0] * 1.5))))
	if cv2.waitKey(5) & 0xFF == 27:
		break
cap.release()