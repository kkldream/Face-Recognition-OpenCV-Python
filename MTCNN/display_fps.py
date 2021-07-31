import cv2

class DisplayFPS:
    def __init__(self, valus=0):
        self.fps = valus
        self.time = cv2.getTickCount()

    def count(self):
        period = (cv2.getTickCount() - self.time) / cv2.getTickFrequency()
        self.fps = 1.0 / period
        self.time = cv2.getTickCount()

    def print(self, image):
        str = f'FPS:{self.fps:.2f}'
        cv2.putText(image, str, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)