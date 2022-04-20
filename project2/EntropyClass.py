import numpy as np
import math

class Entropy:

	def __init__(self) -> None:
		self.N = 100
		self.x_list = np.zeros(self.N)
		self.y_list = np.zeros(self.N)
		self.l_list = np.zeros(self.N)
		self.Fxy_c_list_size = 300
		self.Fxy_c_list = [(0, 0)] * self.Fxy_c_list_size

	def __call__(self, face_result, info=False):
		return self.get_entropy(face_result, info)

	def set_shape(self, shape):
		self.shape = shape

	def get_entropy(self, face_result, info=False):
		info_dict = dict()
		''' FFT '''
		Ax = int((face_result[33][:2][0] + face_result[133][:2][0]) / 2)
		Ay = int((face_result[33][:2][1] + face_result[133][:2][1]) / 2)
		Bx = int((face_result[362][:2][0] + face_result[263][:2][0]) / 2)
		By = int((face_result[362][:2][1] + face_result[263][:2][1]) / 2)
		Cx = int((face_result[76][:2][0] + face_result[306][:2][0]) / 2)
		Cy = int((face_result[76][:2][1] + face_result[306][:2][1]) / 2)
		if info:
			info_dict['FFT'] = [(Ax, Ay), (Bx, By), (Cx, Cy)]
		''' FFV '''
		S_0 = 2500
		col, row = self.shape[1], self.shape[0]
		Fx = (Ax + Bx + Cx) / 3
		Fy = (Ay + By + Cy) / 3
		S = abs(Ax * By - Bx * Ay + Bx * Cy - Cx * By + Cx * Ay - Ax * Cy) / 2
		x = (Fx - col / 2) * ((S / S_0) ** 0.5) + col / 2
		y = (Fy - row / 2) * ((S / S_0) ** 0.5) + row / 2
		if info:
			info_dict['FFV'] = [(int(x), int(y))]
			info_dict['S_0'] = S_0
			info_dict['S'] = S
		''' Entropy '''
		self.arrAppSlide(self.x_list, x)
		self.arrAppSlide(self.y_list, y)
		Fx_c = np.sum(self.x_list) / self.N
		Fy_c = np.sum(self.y_list) / self.N
		self.arrAppSlide(self.Fxy_c_list, (int(Fx_c), int(Fy_c)))
		if info:
			info_dict['Fxy_c_list'] = self.Fxy_c_list
		l = ((Fx - Fx_c) ** 2 + (Fy - Fy_c) ** 2) ** 0.5
		self.arrAppSlide(self.l_list, l)
		mv = np.sum(self.l_list) / self.N
		sd = (np.sum((self.l_list - mv) ** 2) / self.N) ** 0.5
		i_max = int(np.max(self.l_list) / (mv / sd)) + 1
		I_list = np.array([i * (mv / sd) for i in range(i_max + 2)])
		H_f = 0
		for I in I_list:
			p_i = I / self.N
			if p_i != 0: log_p = math.log(p_i, 2)
			else: log_p = 0
			H_f -= p_i * log_p
		if info: 
			return H_f, info_dict
		return H_f

	def arrAppSlide(self, arr, var):
		arr[1:] = arr[:-1]
		arr[0] = var
