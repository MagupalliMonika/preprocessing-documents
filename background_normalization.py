# pipeline/background_normalization.py

import cv2
import numpy as np


class BackgroundNormalization:

    def __init__(
        self,
        brightness_threshold=200,
        color_diff_threshold=15
    ):
        self.brightness_threshold = brightness_threshold
        self.color_diff_threshold = color_diff_threshold

    # ----------------------------------------
    # Step 1: Detect color cast
    # ----------------------------------------
    def detect_color_cast(self, image):

        bright_mask = np.all(
            image > self.brightness_threshold,
            axis=2
        )

        bright_pixels = image[bright_mask]

        if bright_pixels.size == 0:
            return False, None, 0.0

        mean_color = np.mean(bright_pixels, axis=0)

        diff = np.linalg.norm(
            mean_color - np.array([255, 255, 255])
        )

        has_cast = diff > self.color_diff_threshold

        return has_cast, mean_color, diff

    # ----------------------------------------
    # Step 2: Normalize if needed
    # ----------------------------------------
    def apply(self, image):

        has_cast, mean_color, diff = self.detect_color_cast(image)

        if not has_cast:
            return image, {
                "needed": False,
                "deviation": float(diff),
                "mean_background_color": None if mean_color is None else mean_color.tolist()
            }

        # LAB correction
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        a_blur = cv2.GaussianBlur(a, (21, 21), 0)
        b_blur = cv2.GaussianBlur(b, (21, 21), 0)

        a_corrected = cv2.addWeighted(a, 1.0, a_blur, -1.0, 128)
        b_corrected = cv2.addWeighted(b, 1.0, b_blur, -1.0, 128)

        lab_corrected = cv2.merge([l, a_corrected, b_corrected])

        normalized_img = cv2.cvtColor(
            lab_corrected,
            cv2.COLOR_LAB2BGR
        )

        normalized_img = np.clip(
            normalized_img,
            0,
            255
        ).astype(np.uint8)

        return normalized_img, {
            "needed": True,
            "deviation": float(diff),
            "mean_background_color": mean_color.tolist()
        }