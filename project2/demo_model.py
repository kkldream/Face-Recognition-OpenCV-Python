import cv2
import numpy as np
import utils
import time
from face_mesh import FaceMesh
import json
from keras.models import load_model
from EntropyClass import Entropy

# my_model = load_model('models/2022_04_19-14_54_12-Conv2-params_36353-batch_256-optimizer_adam-loss_mse/epoch_065-val_loss_0.002.hdf5')
my_model = load_model('model.hdf5')


def main():
    level = 0
    mar = 0
    H_f = 0
    TestList = np.zeros((300, 2))
    ''' CaptureInput '''
    # cap = utils.CaptureInput(0, 640, 480, 30)
    cap = utils.CaptureInput('../dataset/YawDD/test/1-MaleGlasses.avi')
    # cap = utils.CaptureInput('blink_test.mp4', 960, 540)
    cap.setFlip = True
    ''' Create object '''
    face_mesh = FaceMesh(1, 0.7, 0.7)
    cvFpsCalc = utils.CvFpsCalc(buffer_len=10)
    entropy = Entropy()
    ''' Start Loop '''
    while True:
        display_fps = cvFpsCalc.get()
        ret, frame = cap.read()
        entropy.set_shape(frame.shape)
        face_results = face_mesh(frame)
        for face_result in face_results:
            ''' mouth detection '''
            mar = calc_mar(face_result)
            H_f = entropy(face_result)
            draw_mouth_edge(frame, face_result)
            draw_eye_edge(frame, face_result)
            Test = np.array([
                mar,
                map(H_f, 0, 30, 0, 1)
            ])
            TestList[:-1] = TestList[1:]
            TestList[-1] = Test
            predicted = my_model.predict([
                np.array([TestList[:, 0]]),
                np.array([TestList[-1, 1]])
            ])
            level = max(0, min(1, predicted[0, 0]))
        ''' display '''
        cv2.putText(frame, "FPS:" + str(display_fps), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f'Level:{int(level * 100)}%', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "mar:" + f'{mar:.2f}', (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "H_f:" + f'{H_f:.2f}', (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
    cap.release()
    cv2.destroyAllWindows()


def calc_ear(landmarks):
    p1 = landmarks[33]
    p2 = landmarks[160]
    p3 = landmarks[158]
    p4 = landmarks[133]
    p5 = landmarks[153]
    p6 = landmarks[144]
    left_ear = (utils.get_distance(p2, p6) + utils.get_distance(p3, p5)) \
               / (2 * utils.get_distance(p1, p4))
    p1 = landmarks[263]
    p2 = landmarks[387]
    p3 = landmarks[385]
    p4 = landmarks[362]
    p5 = landmarks[380]
    p6 = landmarks[373]
    right_ear = (utils.get_distance(p2, p6) + utils.get_distance(p3, p5)) \
                / (2 * utils.get_distance(p1, p4))
    return left_ear, right_ear


def calc_mar(landmarks):
    p1 = landmarks[61]
    p2 = landmarks[81]
    p3 = landmarks[311]
    p4 = landmarks[291]
    p5 = landmarks[402]
    p6 = landmarks[178]
    ear = (utils.get_distance(p2, p6) + utils.get_distance(p3, p5)) \
          / (2 * utils.get_distance(p1, p4))
    return ear


def draw_mouth_edge(image, landmarks):
    pos = (61, 81, 311, 291, 402, 178)
    for p in pos:
        cv2.circle(image, landmarks[p][:2], 2, (0, 0, 255), -1)


def draw_eye_edge(image, landmarks):
    pos = [
        landmarks[33],
        landmarks[160],
        landmarks[158],
        landmarks[133],
        landmarks[153],
        landmarks[144],
        landmarks[263],
        landmarks[387],
        landmarks[385],
        landmarks[362],
        landmarks[380],
        landmarks[373]
    ]
    for p in pos:
        cv2.circle(image, p[:2], 1, (0, 0, 255), -1)


def map(var, in_min, in_max, out_min, out_max):
    var = float(var)
    if var >= in_max:
        return out_max
    if var <= in_min:
        return out_min
    return (var - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


if __name__ == '__main__':
    main()
