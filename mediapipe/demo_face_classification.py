import cv2
import numpy as np
import utils
import time
from face_mesh import FaceMesh
from face_classification import FaceClassification

def main():
    ''' CaptureInput '''
    cap = utils.CaptureInput(0, 640, 480, 30)
    cap.setFlip = True
    ''' Create object '''
    face_mesh = FaceMesh(1, 0.7, 0.7)
    cvFpsCalc = utils.CvFpsCalc(buffer_len=10)
    face_classification = FaceClassification()
    head_wait_time = 0
    head_last_pos = (0, 0)
    ''' Start Loop'''
    name, credibility = 0, 0
    while True:
        display_fps = cvFpsCalc.get()
        ret, frame = cap.read()
        face_results = face_mesh(frame)
        for face_result in face_results:
            ''' face bbox '''
            face_bbox = face_mesh.calc_face_bbox(face_result)
            # cv2.rectangle(frame, face_bbox[0], face_bbox[1], (255,0,0), 2)
            ''' get rotation '''
            roll, yaw, pitch = face_mesh.get_rotation(face_result)
            ''' get align roi '''
            face_roi = face_mesh.get_align_roi(frame, face_result, roll)
            face_roi = face_classification.get_resize(face_roi)
            cv2.imshow('face_roi', face_roi)
            ''' face classification '''
            head_pos = face_mesh.calc_face_mid(face_result)
            move_rate = utils.get_distance(head_pos, head_last_pos) / (face_bbox[1][0] - face_bbox[0][0])
            if time.time() - head_wait_time > 10 or move_rate > 0.5:
                head_wait_time = time.time()
                head_last_pos = head_pos
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
