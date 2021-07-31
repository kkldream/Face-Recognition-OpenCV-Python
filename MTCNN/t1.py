from mtcnn import MTCNN
from display_fps import DisplayFPS
import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 288)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 352)
mtcnn = MTCNN()
fps = DisplayFPS(cap.get(cv2.CAP_PROP_FPS))

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    det = mtcnn.detect(frame)
    for d in det:
        score, box, landmark = d
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 1)
        cv2.putText(frame, '{:.3f}'.format(score), (box[0], box[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        for j in range(5):
            cv2.circle(frame, (int(landmark[2*j]),int(int(landmark[2*j+1]))), 2, (0,0,255))
    fps.count()
    fps.print(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
