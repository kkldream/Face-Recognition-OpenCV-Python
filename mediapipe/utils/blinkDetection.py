import math

EAR_LIST = [[133, 158, 160, 33, 144, 153],
            [362, 385, 387, 263, 373, 380]]

def _get_distance(point_0, point_1):
    distance = math.pow((point_0[0] - point_1[0]), 2) + math.pow((point_0[1] - point_1[1]), 2)
    distance = math.sqrt(distance)
    return distance

def get_ear_list(eye_landmarks):
    left_ear_list = [eye_landmarks[i][:2] for i in EAR_LIST[0]]
    right_ear_list = [eye_landmarks[i][:2] for i in EAR_LIST[1]]
    return left_ear_list, right_ear_list

def get_ear(ear_list):
    a = _get_distance(ear_list[1], ear_list[5])
    b = _get_distance(ear_list[2], ear_list[4])
    c = 2 * _get_distance(ear_list[0], ear_list[3])
    ear = (a + b) / c
    return ear
