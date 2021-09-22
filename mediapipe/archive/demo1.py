import cv2
import numpy as np
import utils
from face_mesh import FaceMesh
import matplotlib.pyplot as plt

def main():
    cap = utils.CaptureInput(0, 640, 480, 30)
    # cap = utils.CaptureInput('blink_test.mp4', 960, 540)
    cap.setFlip = True
    face_mesh = FaceMesh(1, 0.7, 0.7)
    cvFpsCalc = utils.CvFpsCalc(buffer_len=10)
    while True:
        display_fps = cvFpsCalc.get()
        ret, frame = cap.read()
        face_results = face_mesh(frame)
        for face_result in face_results:
            left_eye_landmarks, right_eye_landmarks = face_mesh.get_eye_landmarks(face_result)
            # left_eye_landmarks = np.array(left_eye_landmarks).reshape(-1, 1, 2)
            # cv2.drawContours(frame, left_eye_landmarks, -1, (255, 0, 0), 3)
            # right_eye_landmarks = np.array(right_eye_landmarks).reshape(-1, 1, 2)
            # cv2.drawContours(frame, right_eye_landmarks, -1, (255, 0, 0), 3)
            
            left_ear_list, right_ear_list = utils.get_ear_list(face_result)
            # left_ear_list = np.array(left_ear_list).reshape(-1, 1, 2)
            # cv2.drawContours(frame, left_ear_list, -1, (0, 0, 255), 3)
            left_ear = utils.get_ear(left_ear_list)
            right_ear = utils.get_ear(right_ear_list)
            print(left_ear)

            # for face_landmarks in face_result:
            #     pos = face_landmarks[:2]
            #     cv2.circle(frame, pos, 1, (0, 0, 255), -1)
        cv2.putText(frame, "FPS:" + str(display_fps), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f'left_ear: {left_ear:.2f}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, f'right_ear: {right_ear:.2f}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == 27: # ESC
            break
    cap.release()
    cv2.destroyAllWindows()
    return

if __name__ == '__main__':
    main()
