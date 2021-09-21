import cv2
import numpy as np
import utils
import time
from face_mesh import FaceMesh
from face_classification import FaceClassification

def main():
    ''' CaptureInput '''
    cap = utils.CaptureInput(0, 640, 480, 30)
    # cap = utils.CaptureInput('blink_test.mp4', 960, 540)
    cap.setFlip = True
    ''' Create object '''
    face_mesh = FaceMesh(1, 0.7, 0.7)
    cvFpsCalc = utils.CvFpsCalc(buffer_len=10)
    face_classification = FaceClassification()
    classification_time = 0
    classification_last_pos = (0, 0)
    ''' Start Loop'''
    name, credibility = 0, 0
    while True:
        display_fps = cvFpsCalc.get()
        ret, frame = cap.read()
        face_results = face_mesh(frame)
        for face_result in face_results:
            ''' eyes contours'''
            # left_eye_landmarks, right_eye_landmarks = face_mesh.get_eye_landmarks(face_result)
            # left_eye_landmarks = np.array(left_eye_landmarks).reshape(-1, 1, 2)
            # cv2.drawContours(frame, left_eye_landmarks, -1, (255, 0, 0), 3)
            # right_eye_landmarks = np.array(right_eye_landmarks).reshape(-1, 1, 2)
            # cv2.drawContours(frame, right_eye_landmarks, -1, (255, 0, 0), 3)
            ''' face landmarks '''
            # for face_landmark in face_result:
            #     pos = face_landmark[:2]
            #     cv2.circle(frame, pos, 1, (0, 0, 255), -1)
            ''' face bbox '''
            face_bbox = face_mesh.calc_face_bbox(face_result)
            # cv2.rectangle(frame, face_bbox[0], face_bbox[1], (255,0,0), 2)
            ''' mtcnn landmarks '''
            # mtcnn_landmarks = face_mesh.get_mtcnn_landmarks(face_result)
            # mtcnn_landmarks = np.array(mtcnn_landmarks).reshape(-1, 1, 2)
            # cv2.drawContours(frame, mtcnn_landmarks, -1, (0, 0, 255), 3)
            ''' get rotation '''
            roll, yaw, pitch = face_mesh.get_rotation(face_result)
            ''' get align roi '''
            face_roi = face_mesh.get_align_roi(frame, face_result, roll)
            face_roi = face_classification.get_resize(face_roi)
            # print(f'{roll}, {yaw}, {pitch}')
            cv2.imshow('face_roi', face_roi)
            ''' face classification '''
            classification_pos = face_mesh.calc_face_mid(face_result)
            move_rate = utils.get_distance(classification_pos, classification_last_pos) / (face_bbox[1][0] - face_bbox[0][0])
            if time.time() - classification_time > 10 or move_rate > 0.5:
                classification_time = time.time()
                classification_last_pos = classification_pos
                name, credibility = face_classification(face_roi, roll)
            ''' display '''
            cv2.rectangle(frame, face_bbox[0], face_bbox[1], (255, 0, 0))
            cv2.putText(frame, f'{name} - {int(credibility)}%', face_bbox[0],
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, "FPS:" + str(display_fps), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == 27: # ESC
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
