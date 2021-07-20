import cv2
import math
import pcn

def draw_rectangle(image, vertex_list):
    image_copy = image.copy()
    for vertex in vertex_list:
        cv2.line(image_copy, vertex[0], vertex[1], (255, 0, 0), 2)
        cv2.line(image_copy, vertex[1], vertex[2], (0, 0, 255), 2)
        cv2.line(image_copy, vertex[2], vertex[3], (0, 0, 255), 2)
        cv2.line(image_copy, vertex[3], vertex[0], (0, 0, 255), 2)
        for i in vertex:
            cv2.circle(image_copy, i, 5, (0, 255, 0), -1)
    return image_copy

def get_vertex(win_list):
    vertex_list = []
    for face in win_list:
        mid = (face.x + face.width / 2, face.y + face.width / 2)
        r = face.width / 2 * (2 ** (1 / 2))
        vertex = []
        for i in range(4):
            angle = 0
            if face.angle != 45 + 90 * i:
                angle = math.pi / (180 / (face.angle - (45 + 90 * i)))
            vertex.append((int(mid[0] - r * math.cos(angle * -1)),
                           int(mid[1] - r * math.sin(angle * -1))))
        vertex_list.append(vertex)
    return vertex_list

def rotate_crop(image, face):
    mid = (face.x + face.width / 2, face.y + face.width / 2)
    width = int(face.width * (2 ** (1 / 2)))
    vertex = (int(mid[0] - width / 2), int(mid[1] - width / 2))
    dst = image[vertex[1]:vertex[1] + width, vertex[0]:vertex[0] + width]
    matrix = cv2.getRotationMatrix2D((width / 2, width / 2), face.angle * -1, 1)
    dst = cv2.warpAffine(dst, matrix, (width, width))
    w = int((width - face.width) / 2)
    dst = dst[w:w + face.width, w:w + face.width]
    return dst

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        win_list = pcn.detect(frame)
        img_rect = pcn.draw(frame, win_list)
        if len(win_list) > 0:
            img_roi = rotate_crop(frame, win_list[0])
            cv2.imshow('ROI', img_roi)
        cv2.imshow('PCN', img_rect)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
