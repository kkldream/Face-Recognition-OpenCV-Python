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

    def get_align_roi(self, image, landmarks, roll=False):
        face_top = landmarks[10]
        face_bottom = landmarks[152]
        face_left = landmarks[234]
        face_right = landmarks[454]
        face_mid = (int((face_top[0] + face_bottom[0]) / 2), int((face_left[1] + face_right[1]) / 2))
        left_top = (int(face_left[0] - (face_mid[0] - face_top[0])), int(face_left[1] - (face_mid[1] - face_top[1])))
        right_top = (int(face_right[0] - (face_mid[0] - face_top[0])), int(face_right[1] - (face_mid[1] - face_top[1])))
        left_bottom = (int(face_left[0] - (face_mid[0] - face_bottom[0])), int(face_left[1] - (face_mid[1] - face_bottom[1])))
        right_bottom = (int(face_right[0] - (face_mid[0] - face_bottom[0])), int(face_right[1] - (face_mid[1] - face_bottom[1])))
        roi_mid = int(max((right_bottom[0] + left_top[0]) / 2, (right_top[0] + left_bottom[0]) / 2)), int(max((left_bottom[1] + right_top[1]) / 2, right_bottom[1] + left_top[1]) / 2)
        roi_wight = max(utils.get_distance(right_bottom, left_top), utils.get_distance(right_top, left_bottom))
        roi_hidght = max(utils.get_distance(left_bottom, left_top), utils.get_distance(right_bottom, right_top))
        roi_max_side = max(roi_wight, roi_hidght)
        if roll is False:
            roll = self.get_rotation(landmarks)[0]
        frame = utils.rotate_img(image, roll, roi_mid)
        boxs = (
            utils.rotate_xy(left_top, roi_mid, roll),
            utils.rotate_xy(right_top, roi_mid, roll),
            utils.rotate_xy(right_bottom, roi_mid, roll),
            utils.rotate_xy(left_bottom, roi_mid, roll)
        )
        max_side = max(
            boxs[2][1] - boxs[0][1],
            max(boxs[1][0], boxs[2][0]) - min(boxs[0][0], boxs[3][0])
        )
        # face_roi = frame.copy()[
        #     boxs[0][1]:boxs[2][1],
        #     min(boxs[0][0], boxs[3][0]):max(boxs[1][0], boxs[2][0])
        # ]
        
        # troi = (boxs[0][1], min(boxs[0][0], boxs[3][0]))
        # face_roi = frame.copy()[
        #     troi[0]:troi[0] + max_side,
        #     troi[1]:troi[1] + max_side
        # ]
        face_roi = frame.copy()[
            int(roi_mid[1] - max_side / 2):int(roi_mid[1] + max_side / 2),
            int(roi_mid[0] - max_side / 2):int(roi_mid[0] + max_side / 2)
        ]
        face_roi = cv2.resize(face_roi, (160, 160))
        return face_roi

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