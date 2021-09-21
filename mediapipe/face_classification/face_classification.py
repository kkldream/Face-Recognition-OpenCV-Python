import json
import utils
import numpy as np
import cv2

from face_classification.face_feature import FaceFeature
from face_classification.tf_graph import FaceRecGraph

class FaceClassification:
    def __init__(self) -> None:
        self.extract_feature = FaceFeature(FaceRecGraph())
        f = open('face_classification/facerec_128D.txt','r')
        self.data_set = json.loads(f.read())

    def __call__(self, image, roll):
        position = self._get_pos(roll)
        feature = self.extract_feature.get_features([image])
        # self._findPeople(feature[0])
        name, credibility = self._findPeople(feature[0], position)
        return name, credibility

    def _get_pos(self, roll):
        if roll > 30:
            return "Right"
        elif roll < -30:
            return "Left"
        return "Center"
        # points = self._get_mtcnn_landmarks(landmarks)
        # if abs(points[0][0] - points[1][0]) / abs(points[0][1] - points[1][0]) > 2:
        #     return "Right"
        # elif abs(points[0][1] - points[1][0]) / abs(points[0][0] - points[1][0]) > 2:
        #     return "Left"
        # return "Center"

    def get_resize(self, image, size=160):
        image = cv2.resize(image, (160, 160))
        return image

    def _findPeople(self, features, position, thres = 0.6, percent_thres = 70):
        '''
        :param features_arr: a list of 128d Features of all faces on screen
        :param positions: a list of face position types of all faces on screen
        :param thres: distance threshold
        :return: person name and percentage
        '''
        data_set = self.data_set
        result = "Unknown"
        smallest = -1
        for person in data_set.keys():
            person_data = data_set[person][position]
            for data in person_data:
                distance = np.sqrt(np.sum(np.square(data - features)))
                if(distance < smallest or smallest == -1):
                    smallest = distance
                    result = person
        percentage =  min(100, 100 * thres / smallest)
        if percentage <= percent_thres :
            result = "Unknown"
        return result, percentage