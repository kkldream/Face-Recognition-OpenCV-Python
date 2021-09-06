face_mark = {
    'left_eye_top' : 159,
    'left_eye_bottom' : 145,
    'right_eye_top' : 386,
    'right_eye_bottom' : 374,
    'mouth_top' : 13,
    'mouth_bottom' : 14,
    'face_top' : 10,
    'face_bottom' : 152,
    'face_left' : 127,
    'face_right' : 356,
    'left_eye_left_corner' : 226,
    'left_eye_right_corner' : 243,
    'right_eye_left_corner' : 463,
    'right_eye_right_corner' : 446,
    'between_mouth_nose' : 164
}

class FaceLandmark():
    def __init__(self, face_landmarks, image_shape) -> None:
        self._face_landmarks = face_landmarks
        self._image_shape = image_shape

    def get_coordinates(self, face_mark_key):
        mark_num = face_mark[face_mark_key]
        landmark = self._face_landmarks.landmark[mark_num]
        x = int(self._image_shape[0] * landmark.x)
        y = int(self._image_shape[1] * landmark.y)
        z = landmark.z
        return x, y, z