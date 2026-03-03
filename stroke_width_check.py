# pipeline/stroke_width_check.py

import cv2
import numpy as np
from skimage.morphology import skeletonize


class StrokeWidthConsistency:

    def __init__(self, deviation_threshold=0.3):
        self.deviation_threshold = deviation_threshold

    def evaluate(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Binarize
        _, binary = cv2.threshold(
            gray,
            0,
            1,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Skeletonize
        skeleton = skeletonize(binary).astype(np.uint8)

        # Distance transform
        dist_transform = cv2.distanceTransform(
            binary.astype(np.uint8),
            cv2.DIST_L2,
            3
        )

        # Stroke widths
        stroke_widths = 2 * dist_transform[skeleton == 1]

        if len(stroke_widths) == 0:
            return {
                "valid": False,
                "mean_width": 0.0,
                "std_width": 0.0,
                "deviation_ratio": 0.0,
                "reason": "No text detected"
            }

        mean_width = float(np.mean(stroke_widths))
        std_width = float(np.std(stroke_widths))
        deviation_ratio = std_width / mean_width if mean_width > 0 else 0

        is_consistent = deviation_ratio < self.deviation_threshold

        return {
            "valid": is_consistent,
            "mean_width": mean_width,
            "std_width": std_width,
            "deviation_ratio": deviation_ratio,
            "threshold": self.deviation_threshold
        }