import cv2
import numpy as np


class UnevenIlluminationCorrection:

    def __init__(self, alpha=0.8, blur_kernel=(101, 101)):
        self.alpha = alpha
        self.blur_kernel = blur_kernel

    def apply(self, image):

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        background = cv2.GaussianBlur(
            l,
            self.blur_kernel,
            0
        ).astype(np.float32)

        l_float = l.astype(np.float32)

        l_normalized = (l_float / (background + 1e-5)) * np.mean(background)
        l_normalized = np.clip(l_normalized, 0, 255)

        l_corrected = (
            self.alpha * l_normalized +
            (1 - self.alpha) * l_float
        )

        l_corrected = np.clip(l_corrected, 0, 255).astype(np.uint8)

        lab_corrected = cv2.merge([l_corrected, a, b])
        corrected = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)

        return corrected