import cv2
import numpy as np

class KalmanFilter:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4,2)
        self.kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
        self.kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
        self.kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.03
        # self.kalman.measurementNoiseCov = np.array([[1,0],[0,1]],np.float32) * 0.00003
    
    def correct(self, mp):
        self.kalman.correct(mp)
    
    def predict(self):
        tp = self.kalman.predict()
        return (tp[0][0], tp[1][0])
        
if __name__ == '__main__':
    frame = np.zeros((400,300,3), np.uint8) # drawing canvas
    last_mp = np.zeros(2, np.int32)
    last_tp = np.zeros(2, np.int32)

    mp = np.zeros(2, np.float32)
    def onmouse(k,x,y,s,p):
        global mp
        mp = np.array((x, y), dtype='float32')


    cv2.namedWindow("kalman")
    cv2.setMouseCallback("kalman",onmouse)

    kalman = KalmanFilter()
    while True:
        size = frame.shape
        mp_r = np.array((mp[0] / size[1], mp[1] / size[0]), dtype=np.float32)
        kalman.correct(mp_r)
        tp_r = kalman.predict()
        tp = (int(size[1] * tp_r[0]), int(size[0] * tp_r[1]))
        cv2.line(frame, last_mp, mp.astype(int), (0,100,0))
        cv2.line(frame, last_tp, tp, (0,0,200))
        last_mp = mp.copy().astype(int)
        last_tp = np.array(tp)
        cv2.imshow("kalman",frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break