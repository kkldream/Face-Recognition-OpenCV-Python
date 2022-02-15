import cv2
import numpy as np
import utils
from face_mesh import FaceMesh
import math

def main():
	cap = utils.CaptureInput(0, 640, 480, 30)
	cap.setFlip = True
	face_mesh = FaceMesh(1, 0.7, 0.7)
	cvFpsCalc = utils.CvFpsCalc(buffer_len=10)
	N = 100
	x_list = np.zeros(N)
	y_list = np.zeros(N)
	l_list = np.zeros(N)
	while True:
		display_fps = cvFpsCalc.get()
		ret, frame = cap.read()
		face_results = face_mesh(frame)
		for face_result in face_results:
			''' FFT '''
			Ax = int((face_result[33][:2][0] + face_result[133][:2][0]) / 2)
			Ay = int((face_result[33][:2][1] + face_result[133][:2][1]) / 2)
			Bx = int((face_result[362][:2][0] + face_result[263][:2][0]) / 2)
			By = int((face_result[362][:2][1] + face_result[263][:2][1]) / 2)
			Cx = int((face_result[76][:2][0] + face_result[306][:2][0]) / 2)
			Cy = int((face_result[76][:2][1] + face_result[306][:2][1]) / 2)
			cv2.line(frame, (Ax, Ay), (Bx, By), (255, 0, 0), 2)
			cv2.line(frame, (Ax, Ay), (Cx, Cy), (255, 0, 0), 2)
			cv2.line(frame, (Bx, By), (Cx, Cy), (255, 0, 0), 2)
			[cv2.circle(frame, p, 3, (0, 0, 255), -1) for p in [(Ax, Ay), (Bx, By), (Cx, Cy)]]
			''' FFTs '''
			S_0 = 1800
			col, row = frame.shape[1], frame.shape[0]
			Fx = int((Ax + Bx + Cx) / 3)
			Fy = int((Ay + By + Cy) / 3)
			S = abs(Ax * By - Bx * Ay + Bx * Cy - Cx * By + Cx * Ay - Ax * Cy) / 2
			x = int((Fx - col / 2) * ((S / S_0) ** 0.5) + col / 2)
			y = int((Fy - row / 2) * ((S / S_0) ** 0.5) + row / 2)
			cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
			''' FFV '''
			arrAppSlide(x_list, x)
			arrAppSlide(y_list, y)
			Fx_c = np.sum(x_list) / N
			Fy_c = np.sum(y_list) / N
			drawPoint(frame, (Fx_c, Fy_c))
			l = ((Fx - Fx_c) ** 2 + (Fy - Fy_c) ** 2) ** 0.5
			arrAppSlide(l_list, l)
			mv = np.sum(l_list) / N
			sd = (np.sum((l_list - mv) ** 2) / N) ** 0.5
			i_max = int(np.max(l_list) / (mv / sd)) + 1
			I_list = np.array([i * (mv / sd) for i in range(i_max + 2)])
			H_f = 0
			for I in I_list:
				p_i = I / N
				if p_i != 0: log_p = math.log(p_i, 2)
				else: log_p = 0
				H_f -= p_i * log_p
			cv2.putText(frame, f'H_f = {H_f}', (10, 60),
						cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

		cv2.putText(frame, "FPS:" + str(display_fps), (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
		cv2.imshow('frame', frame)
		if cv2.waitKey(1) == 27: # ESC
			break
	cap.release()
	cv2.destroyAllWindows()
	return

def drawPoint(frame, point, str=''):
	cv2.circle(frame, (int(point[0]), int(point[1])), 3, (255, 0, 0), -1)
	cv2.putText(frame, str, (int(point[0]), int(point[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

def arrAppSlide(arr, var):
	arr[1:] = arr[:-1]
	arr[0] = var

if __name__ == '__main__':
	main()
