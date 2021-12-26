import cv2
import numpy as np
import utils        
        
kalman = utils.KalmanFilter(1, 0.01)

plt_1 = utils.PltOpenCV(100)
plt_2 = utils.PltOpenCV(100)
last_measurement = 0
while True:
    measurement = [last_measurement + np.random.randint(-10, 10)]
    measurement[0] = min(100, max(0, measurement[0]))
    last_measurement = measurement[0]
    prediction = kalman(measurement)
    print(prediction)
    plt_1_img = plt_1(measurement[0])
    plt_2_img = plt_2(prediction[0])
    cv2.imshow("plt_1_img",plt_1_img)
    cv2.imshow("plt_2_img",plt_2_img)
    if (cv2.waitKey(1) & 0xff) == 27:
        break
cv2.destroyAllWindows()