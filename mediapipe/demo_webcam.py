import cv2
import utils

def main():
    ''' CaptureInput '''
    cap = utils.CaptureInput(0, 640, 480, 15)
    cap.setFlip = True
    ''' Create object '''
    cvFpsCalc = utils.CvFpsCalc(buffer_len=10)
    ''' Start Loop'''
    while True:
        display_fps = cvFpsCalc.get()
        ret, frame = cap.read()
        cv2.putText(frame, "FPS:" + str(display_fps), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == 27: # ESC
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
