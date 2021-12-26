import cv2
import numpy as np
import utils

kalman = cv2.KalmanFilter(4,2)
#設定測量矩陣
kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
#設定轉移矩陣
kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
#設定過程噪聲協方差矩陣
kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32)*0.03

plt_1 = utils.PltOpenCV(100)
plt_2 = utils.PltOpenCV(100)
last_x = 0
while True:
    x = last_x + np.random.randint(-10, 10)
    x = min(100, max(0, x))
    last_x = x
    y = 0
    current_measurement = np.array([[np.float32(x)],[np.float32(y)]]) # 傳遞當前測量座標值
    kalman.correct(current_measurement) # 用來修正卡爾曼濾波的預測結果
    current_prediction = kalman.predict() # 呼叫kalman這個類的predict方法得到狀態的預測值矩陣，用來估算目標位置
    cmx,cmy = current_measurement[0],current_measurement[1] # 當前測量值
    cpx,cpy = current_prediction[0],current_prediction[1] # 當前預測值
    plt_1_img = plt_1(cmx)
    plt_2_img = plt_2(cpx)
    cv2.imshow("plt_1_img",plt_1_img)
    cv2.imshow("plt_2_img",plt_2_img)
    if (cv2.waitKey(30) & 0xff) == 27:
        break
cv2.destroyAllWindows()