import dlib
import cv2

predictor_path = "models/shape_predictor_5_face_landmarks.dat"

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def renderFace(im, landmarks, color=(0, 255, 0), radius=3):
  for p in landmarks.parts():
    cv2.circle(im, (p.x, p.y), radius, color, -1)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    dets = detector(frame, 1)
    for k, d in enumerate(dets):
        shape = predictor(frame, d)
        renderFace(frame, shape)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()