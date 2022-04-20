import cv2
import utils
from face_mesh import FaceMesh
from Entropy import Entropy

def main():
	# cap = utils.CaptureInput(0, 640, 480, 30)
	# cap.setFlip = True
	cap = utils.CaptureInput("../drive-yawning-detection/dataset/YawDD/test/2-FemaleNoGlasses.avi")
	face_mesh = FaceMesh(1, 0.7, 0.7)
	cvFpsCalc = utils.CvFpsCalc(buffer_len=10)
	entropy = Entropy()
	while True:
		display_fps = cvFpsCalc.get()
		ret, frame = cap.read()
		entropy.set_shape(frame.shape)
		face_results = face_mesh(frame)
		for face_result in face_results:
			H_f, info = entropy(face_result, info=True)
			drawText(frame, f'H_f = {H_f:.2f}', 2)
			FFT = info['FFT']
			cv2.line(frame, FFT[0], FFT[1], (255, 0, 0), 2)
			cv2.line(frame, FFT[0], FFT[2], (255, 0, 0), 2)
			cv2.line(frame, FFT[1], FFT[2], (255, 0, 0), 2)
			[cv2.circle(frame, p, 3, (0, 0, 255), -1) for p in FFT]
			FFV = info['FFV']
			S_0 = info['S_0']
			S = info['S']
			cv2.circle(frame, FFV[0], 5, (0, 0, 255), -1)
			drawText(frame, f'S_0 = {int(S_0)}', 3)
			drawText(frame, f'S = {int(S)}', 4)
			Fxy_c_list = info['Fxy_c_list']
			for i in range(len(Fxy_c_list) - 1, 0, -1):
				cv2.line(frame, Fxy_c_list[i], Fxy_c_list[i - 1], (i, 0, len(Fxy_c_list) - i), 2)
			cv2.circle(frame, Fxy_c_list[0], 5, (255, 0, 0), -1)
		drawText(frame, f'FPS = {display_fps}', 1)
		cv2.imshow('frame', frame)
		if cv2.waitKey(1) == 27: # ESC
			break
	cap.release()
	cv2.destroyAllWindows()
	return

def drawText(frame, str, num):
	cv2.putText(frame, str, (10, 30 * num),
				cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

if __name__ == '__main__':
	main()
