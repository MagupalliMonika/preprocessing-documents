import cv2
import numpy as np
import blur_detector


class BlurDetection:

    def __init__(self, threshold=0.65):
        self.threshold = threshold

    def detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blur_map = blur_detector.detectBlur(gray)
        blur_score = float(np.mean(blur_map))

        is_blurry = blur_score > self.threshold

        return {
            "needed": is_blurry,     # True means STOP
            "score": blur_score,
            "threshold": self.threshold
        }