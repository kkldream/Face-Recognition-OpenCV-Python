import cv2
import numpy as np
import utils
import time
from face_mesh import FaceMesh
from iris_landmark import IrisLandmark

def main():
    ''' CaptureInput '''
    cap = utils.CaptureInput(0, 640, 480, 30)
    # cap = utils.CaptureInput('blink_test.mp4', 960, 540)
    cap.setFlip = True
    ''' Create object '''
    face_mesh = FaceMesh(1, 0.7, 0.7)
    iris_detector = IrisLandmark()
    cvFpsCalc = utils.CvFpsCalc(buffer_len=10)
    ''' Start Loop'''
    while True:
        display_fps = cvFpsCalc.get()
        ret, frame = cap.read()
        face_results = face_mesh(frame)
        for face_result in face_results:
            left_eye, right_eye = face_mesh.calc_around_eye_bbox(face_result)
            left_iris, right_iris = detect_iris(frame, iris_detector, left_eye, right_eye)
            left_center, left_radius = calc_min_enc_losingCircle(left_iris)
            right_center, right_radius = calc_min_enc_losingCircle(right_iris)
            debug_image = draw_debug_image(
                debug_image,
                left_iris,
                right_iris,
                left_center,
                left_radius,
                right_center,
                right_radius,
            )
        cv2.putText(frame, "FPS:" + str(display_fps), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == 27: # ESC
            break
    cap.release()
    cv2.destroyAllWindows()

def detect_iris(image, iris_detector, left_eye, right_eye):
    image_width, image_height = image.shape[1], image.shape[0]
    input_shape = iris_detector.get_input_shape()

    # 左目
    # 目の周辺の画像を切り抜き
    left_eye_x1 = max(left_eye[0], 0)
    left_eye_y1 = max(left_eye[1], 0)
    left_eye_x2 = min(left_eye[2], image_width)
    left_eye_y2 = min(left_eye[3], image_height)
    left_eye_image = image[left_eye_y1:left_eye_y2, left_eye_x1:left_eye_x2].copy()
    # 虹彩検出
    eye_contour, iris = iris_detector(left_eye_image)
    # 座標を相対座標から絶対座標に変換
    left_iris = calc_iris_point(left_eye, eye_contour, iris, input_shape)

    # 右目
    # 目の周辺の画像を切り抜き
    right_eye_x1 = max(right_eye[0], 0)
    right_eye_y1 = max(right_eye[1], 0)
    right_eye_x2 = min(right_eye[2], image_width)
    right_eye_y2 = min(right_eye[3], image_height)
    right_eye_image = image[right_eye_y1:right_eye_y2, right_eye_x1:right_eye_x2].copy()
    # 虹彩検出
    # 虹彩検出
    eye_contour, iris = iris_detector(right_eye_image)
    # 座標を相対座標から絶対座標に変換
    right_iris = calc_iris_point(right_eye, eye_contour, iris, input_shape)

    return left_iris, right_iris


def calc_iris_point(eye_bbox, eye_contour, iris, input_shape):
    iris_list = []
    for index in range(5):
        point_x = int(iris[index * 3] *
                      ((eye_bbox[2] - eye_bbox[0]) / input_shape[0]))
        point_y = int(iris[index * 3 + 1] *
                      ((eye_bbox[3] - eye_bbox[1]) / input_shape[1]))
        point_x += eye_bbox[0]
        point_y += eye_bbox[1]

        iris_list.append((point_x, point_y))

    return iris_list


def calc_min_enc_losingCircle(landmark_list):
    center, radius = cv2.minEnclosingCircle(np.array(landmark_list))
    center = (int(center[0]), int(center[1]))
    radius = int(radius)

    return center, radius
    
def draw_debug_image(
    debug_image,
    left_iris,
    right_iris,
    left_center,
    left_radius,
    right_center,
    right_radius,
):
    # 虹彩：外接円
    cv2.circle(debug_image, left_center, left_radius, (0, 255, 0), 2)
    cv2.circle(debug_image, right_center, right_radius, (0, 255, 0), 2)

    # 虹彩：ランドマーク
    for point in left_iris:
        cv2.circle(debug_image, (point[0], point[1]), 1, (0, 0, 255), 2)
    for point in right_iris:
        cv2.circle(debug_image, (point[0], point[1]), 1, (0, 0, 255), 2)

    # 虹彩：半径
    cv2.putText(debug_image, 'r:' + str(left_radius) + 'px',
               (left_center[0] + int(left_radius * 1.5),
                left_center[1] + int(left_radius * 0.5)),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
    cv2.putText(debug_image, 'r:' + str(right_radius) + 'px',
               (right_center[0] + int(right_radius * 1.5),
                right_center[1] + int(right_radius * 0.5)),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

    return debug_image

if __name__ == '__main__':
    main()
