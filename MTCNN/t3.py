from numpy.core.records import array
from mtcnn import MTCNN
from display_fps import DisplayFPS
from Kalman_Filter.KalmanFilter import KalmanFilter
import cv2
import numpy as np
import math
cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 288)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 352)
mtcnn = MTCNN()
fps = DisplayFPS(cap.get(cv2.CAP_PROP_FPS))
kalman_box = [KalmanFilter() for i in range(2)]
kalman_landmark = [KalmanFilter() for i in range(5)]
while True:
    ret, frame = cap.read()
    size = frame.shape
    frame = cv2.flip(frame, 1)
    det = mtcnn.detect(frame)
    faces = list()
    noses = list()
    for d in det:
        score, box, landmark = d
        for i in range(2):
            kalman_box[i].correct(np.array((box[2*i], box[2*i+1]), dtype=np.float32))
            box[2*i] = int(kalman_box[i].predict()[0])
            box[2*i+1] = int(kalman_box[i].predict()[1])
        faces.append(frame[box[1]:box[3],box[0]:box[2]])
        # faces.append(frame.copy()[box[1]:box[3],box[0]:box[2]])
        nose_landmark = (landmark[4], landmark[5])
        nose_width = int((box[2] - box[0]) / 2)
        nose_hight = int((box[3] - box[1]) / 4)
        nose_top = (int(nose_landmark[0]-nose_width/2), int(nose_landmark[1]-nose_hight/4))
        noses.append(frame.copy()[nose_top[1]:nose_top[1]+nose_hight,nose_top[0]:nose_top[0]+nose_width])
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 1)
        cv2.putText(frame, '{:.3f}'.format(score), (box[0], box[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        for i in range(5):
            mp = (landmark[2*i], landmark[2*i+1])
            # cv2.circle(frame, np.array(mp, dtype=np.int), 3, (0,255,0), -1)
            mp_r = np.array((mp[0] / size[1], mp[1] / size[0]), dtype=np.float32)
            kalman_landmark[i].correct(mp_r)
            tp_r = kalman_landmark[i].predict()
            tp = (int(size[1] * tp_r[0]), int(size[0] * tp_r[1]))
            cv2.circle(frame, tp, 2, (0,0,255), -1)
        break
    fps.count()
    fps.print(frame)
    for i in range(len(faces)):
        if faces[i].shape[1] != 0:
            faces_resize = cv2.resize(faces[i], (200, int(200/faces[i].shape[1]*faces[i].shape[0])), interpolation=cv2.INTER_CUBIC)
            cv2.imshow('faces', faces_resize)
            noses_resize = cv2.resize(noses[i], (200, int(200/noses[i].shape[1]*noses[i].shape[0])), interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(noses_resize, cv2.COLOR_BGR2GRAY)
            gray[gray<10] = 255
            cv2.imshow('gray', gray)
            ret, threshold = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY)
            cv2.imshow('threshold', threshold)
            # ded = cv2.dilate(threshold, None, iterations=6)
            ded = cv2.erode(threshold, None, iterations=3)
            ded = cv2.dilate(ded, None, iterations=3)
            cv2.imshow('ded', ded)
            contours, hierarchy = cv2.findContours(ded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # for i in range(len(contours)):
            #     if cv2.contourArea(contours[i]) > 200:
            #         cv2.drawContours(noses_resize, contours[i], -1, (0, 0, 255), 3, lineType=cv2.LINE_AA)
            #         x, y, w, h = cv2.boundingRect(contours[i])
            #         mid = (x + w / 2, y + h / 2)
            pos = list()
            for c in contours:
                area = cv2.contourArea(c)
                if area > 100:
                    cv2.drawContours(noses_resize, c, -1, (0, 0, 255), 3, lineType=cv2.LINE_AA)
                    x, y, w, h = cv2.boundingRect(c)
                    mid = (x + w / 2, y + h / 2)
                    pos.append(mid)
            min_pos = [10000, -1, -1]
            for i in range(len(pos)):
                for j in range(len(pos)):
                    if i == j: continue
                    d = (abs(pos[i][0] - pos[j][0]) ** 2 + abs(pos[i][1] - pos[j][1]) ** 2) ** 0.5
                    if d < min_pos[0]:
                        min_pos[0] = d
                        min_pos[1] = i
                        min_pos[2] = j
            if min_pos[0] != 10000:
                cv2.circle(noses_resize, (int(pos[min_pos[1]][0]), int(pos[min_pos[1]][1])), 5, (255,0,0), -1)
                cv2.circle(noses_resize, (int(pos[min_pos[2]][0]), int(pos[min_pos[2]][1])), 5, (255,0,0), -1)
                at = 0
                try:
                    r = (pos[min_pos[1]][0] - pos[min_pos[2]][0]) / (pos[min_pos[1]][1] - pos[min_pos[2]][1])
                    at = math.atan(r) * 180 / math.pi
                    if at > 0: at = 90 - at
                    else: at = (at + 90) * -1
                except ZeroDivisionError: pass
                cv2.putText(frame, f'angle: {int(at)}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            cv2.imshow('noses', noses_resize)
            # canny = cv2.Canny(img, 30, 150)
            # cv2.imshow(f'canny', canny)
        break
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
    # if cv2.waitKey(250) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
