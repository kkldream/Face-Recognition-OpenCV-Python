from math import pow, sqrt

def get_distance(point_0, point_1):
    distance = sqrt(pow((point_0[0] - point_1[0]),2) + pow((point_0[1] - point_1[1]),2))
    return distance