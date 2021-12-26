import cv2
import numpy as np
import utils
from face_mesh import FaceMesh

def main():
	''' CaptureInput '''
	cap = utils.CaptureInput(0, 640, 480, 30)
	cap.setFlip = True
	''' Create object '''
	face_mesh = FaceMesh(1, 0.7, 0.7)
	cvFpsCalc = utils.CvFpsCalc(buffer_len=10)
	while True:
		display_fps = cvFpsCalc.get()
		ret, frame = cap.read()
		face_results = face_mesh(frame)
		for face_result in face_results:
			''' face bbox '''
			face_bbox = face_mesh.calc_face_bbox(face_result)
			cv2.rectangle(frame, face_bbox[0], face_bbox[1], (255, 0, 0))
			''' get rotation '''
			roll, yaw, pitch = face_mesh.get_rotation(face_result)
			# print(f'{roll}, {yaw}, {pitch}')
			cv2.putText(frame, f'roll:{int(roll)}', (10, 60),
						cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
			cv2.putText(frame, f'yaw:{int(yaw)}', (10, 90),
						cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
			cv2.putText(frame, f'pitch:{int(pitch)}', (10, 120),
						cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
		cv2.putText(frame, "FPS:" + str(display_fps), (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
		cv2.imshow('frame', frame)
		if cv2.waitKey(1) == 27: # ESC
			break
	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
