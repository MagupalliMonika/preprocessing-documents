import cv2
import numpy as np


class BlankPageDetection:

    def __init__(self, blank_threshold=0.02):
        self.blank_threshold = blank_threshold

    def detect(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(
            gray,
            200,
            255,
            cv2.THRESH_BINARY_INV
        )

        foreground_pixels = int(np.count_nonzero(binary))
        total_pixels = int(binary.size)

        ratio = foreground_pixels / total_pixels

        is_blank = ratio < self.blank_threshold

        return {
            "needed": is_blank,   # True means STOP
            "foreground_ratio": ratio,
            "threshold": self.blank_threshold
        }