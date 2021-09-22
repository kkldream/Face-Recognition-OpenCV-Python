import cv2
import numpy as np
import utils
from face_mesh import FaceMesh

def main():
	cap = utils.CaptureInput(0, 640, 480, 30)
	cap.setFlip = True
	face_mesh = FaceMesh(1, 0.7, 0.7)
	cvFpsCalc = utils.CvFpsCalc(buffer_len=10)
	while True:
		display_fps = cvFpsCalc.get()
		ret, frame = cap.read()
		face_results = face_mesh(frame)
		for face_result in face_results:
			for face_landmarks in face_result:
				pos = face_landmarks[:2]
				depth = face_landmarks[2]
				if depth >= 0:
					cv2.circle(frame, pos, 1, (0, 0, int(255 * depth * 50)), -1)
				else:
					cv2.circle(frame, pos, 1, (int(255 * depth * -1 * 50), 0, 0), -1)
		cv2.putText(frame, "FPS:" + str(display_fps), (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
		cv2.imshow('frame', frame)
		if cv2.waitKey(1) == 27: # ESC
			break
	cap.release()
	cv2.destroyAllWindows()
	return

if __name__ == '__main__':
	main()
