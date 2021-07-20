import cv2
from time import sleep
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
exposures = [-7, -5, -3, -1]
filenames = ["img_0.033.jpg", "img_0.25.jpg", "img_2.5.jpg", "img_15.jpg"]
cap.read()
for i in range(4):
    cap.set(cv2.CAP_PROP_EXPOSURE, exposures[i])
    sleep(1)
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2.imshow(f'frame-{i}', frame)
    cv2.imwrite(filenames[i], frame)
cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()