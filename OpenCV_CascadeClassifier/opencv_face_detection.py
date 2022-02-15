import cv2

cap = cv2.VideoCapture(0) #讀取影片
cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #載入特徵分類器

while True:
    ret, frame = cap.read() #讀取每一幀

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #色彩轉灰階
    faceRects = cascade.detectMultiScale(grey, scaleFactor = 1.1, minNeighbors = 3) #開始辨識
    for faceRect in faceRects: #依序框起來
        x, y, w, h = faceRect        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('Haar', frame) #顯示照片

    if cv2.waitKey(1) & 0xFF == ord('q'): break #按Q跳出

cap.release()
cv2.destroyAllWindows()