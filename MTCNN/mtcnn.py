from Detection.MtcnnDetector import MtcnnDetector
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from train_models.mtcnn_model import P_Net, R_Net, O_Net
import numpy as np

class MTCNN:
    def __init__(self):
        thresh = [0.9, 0.6, 0.7]
        min_face_size = 24
        stride = 2
        slide_window = False
        detectors = [None, None, None]
        prefix = ['models/PNet_landmark/PNet', 'models/RNet_landmark/RNet', 'models/ONet_landmark/ONet']
        epoch = [18, 14, 16]
        model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
        PNet = FcnDetector(P_Net, model_path[0])
        detectors[0] = PNet
        RNet = Detector(R_Net, 24, 1, model_path[1])
        detectors[1] = RNet
        ONet = Detector(O_Net, 48, 1, model_path[2])
        detectors[2] = ONet
        self.mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size, stride=stride, threshold=thresh, slide_window=slide_window)
    
    def detect(self, image):
        image = np.array(image.copy())
        boxsc, landmarks = self.mtcnn_detector.detect(image)
        result = list()
        for i in range(boxsc.shape[0]):
            score = boxsc[i,4]
            bbox = boxsc[i,:4].astype(int)
            result.append([score, bbox, landmarks[i].astype(int)])
        return result

if __name__ == '__main__':
    mtcnn = MTCNN()