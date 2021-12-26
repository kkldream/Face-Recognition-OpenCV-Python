import cv2
import utils
import numpy as np
from face_mesh import FaceMesh
from eye_detection import EyeDetection
import matplotlib.pyplot as plt
import time

eye_list_len = 10 * 100

def main():
	start_time = time.time()
	''' CaptureInput '''
	cap = utils.CaptureInput(0, 640, 480, 15)
	cap.setFlip = True
	''' Create object '''
	# plt_1 = utils.PltOpenCV(100)
	face_mesh = FaceMesh(1, 0.7, 0.7)
	eye = EyeDetection()
	cvFpsCalc = utils.CvFpsCalc(buffer_len=10)
	head_last_pos = (0, 0)
	eye_list = np.zeros(eye_list_len)
	eye_eff_list = np.zeros(eye_list_len)
	while cap.isOpened():
		display_fps = cvFpsCalc.get()
		ret, frame = cap.read()
		face_results = face_mesh(frame)
		eye_eff_list_temp = False
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
							cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255 * left_percent / 100), 2)
				cv2.putText(frame, f'{right_percent}%', face_result[258][:2],
							cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255 * right_percent / 100), 2)
				# plt_img = plt_1(left_percent)
				# cv2.imshow('plt_img', plt_img)
				eye_eff_list_temp = True
				eye_eff_list[1:] = eye_eff_list[:-1]
				eye_eff_list[0] = 1
				eye_list[1:] = eye_list[:-1]
				eye_list[0] = min(left_percent, right_percent)
				eye_list[eye_list < 40] = 0
		eye_eff_num = sum(eye_eff_list)
		print(f'{eye_eff_num}, {eye_eff_list}')
		reduce_eye_list = list_reduce_directionality(eye_list)
		blink_times = cal_eye_times(reduce_eye_list)
		blink_freq = blink_times / min(100, time.time() - start_time) * 60
		# blink_freq = blink_times * display_fps / eye_list_len / ((eye_eff_num + 1) / eye_list_len) * 3
		if eye_eff_list_temp is False:
			eye_eff_list[1:] = eye_eff_list[:-1]
			eye_eff_list[0] = 0
		plt_img = utils.PltOpenCV.draw_arr(None, reduce_eye_list)
		cv2.imshow('list_reduce_directionality', plt_img)

		cv2.putText(frame, "FPS:" + str(display_fps), (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
		cv2.putText(frame, "Blink:" + str(f'{blink_freq:.2f}'), (10, 60),
					cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
		cv2.putText(frame, "blink_times:" + str(blink_times), (10, 90),
					cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
		cv2.imshow('frame', frame)
		if cv2.waitKey(1) & 0xFF == 27:
			break
	cap.release()
	cv2.destroyAllWindows()

# def cal_eye_times(arr):
#     times = 0
#     up = 0
#     for i, _ in enumerate(arr[:-1]):
#         a, b = arr[i], arr[i + 1]
#         if b - a > 0:
#             if up != 1: times += 1
#             up = 1
#         elif b - a < 0:
#             up = -1
#     return times

def cal_eye_times(arr):
	times = 0
	for i in arr:
		if i == 0:
			times += 1
	return max(0, times - 1)

def list_reduce_directionality(arr):
    times = 0
    up = 0
    new_arr = [i for i in arr]
    pop_arr = []
    for i, _ in enumerate(arr[:-1]):
        a, b = arr[i], arr[i + 1]
        if b - a == 0:
            pop_arr.append(i)
            up == 0
        elif b - a > 0:
            if up != 1: times += 1
            else: pop_arr.append(i)
            up = 1
        elif b - a < 0:
            if up != -1: pass
            else: pop_arr.append(i)
            up = -1
    pop_arr.reverse()
    for i in pop_arr:
        new_arr.pop(i)
    return new_arr

if __name__ == '__main__':
    main()
