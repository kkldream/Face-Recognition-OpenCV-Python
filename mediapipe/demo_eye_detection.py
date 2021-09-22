import cv2
import utils
import numpy as np
from face_mesh import FaceMesh
from eye_detection import EyeDetection

def main():
	''' CaptureInput '''
	cap = utils.CaptureInput(0, 640, 480, 15)
	cap.setFlip = True
	''' Create object '''
	face_mesh = FaceMesh(1, 0.7, 0.7)
	eye = EyeDetection()
	cvFpsCalc = utils.CvFpsCalc(buffer_len=10)
	head_last_pos = (0, 0)
	while cap.isOpened():
		display_fps = cvFpsCalc.get()
		ret, frame = cap.read()
		face_results = face_mesh(frame)
		for face_result in face_results:
			''' face bbox '''
			face_bbox = face_mesh.calc_face_bbox(face_result)
			# cv2.rectangle(frame, face_bbox[0], face_bbox[1], (255,0,0), 2)
			''' detection eye '''
			head_pos = face_mesh.calc_face_mid(face_result)
			move_rate = utils.get_distance(head_pos, head_last_pos) / (face_bbox[1][0] - face_bbox[0][0])
			head_last_pos = head_pos
			if move_rate < 0.01:
				eye.draw_eye_edge(frame, face_result)
				left_percent, right_percent = eye(face_result, display_fps)
				cv2.putText(frame, f'{left_percent}%', face_result[29][:2],
							cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255*left_percent/100), 2)
				cv2.putText(frame, f'{right_percent}%', face_result[258][:2],
							cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255*right_percent/100), 2)
		cv2.putText(frame, "FPS:" + str(display_fps), (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
		cv2.imshow('frame', frame)
		if cv2.waitKey(1) & 0xFF == 27:
			break
	cap.release()
	cv2.destroyAllWindows()
	
if __name__ == '__main__':
    main()
