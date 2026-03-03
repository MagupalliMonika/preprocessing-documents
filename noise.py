

import cv2
import numpy as np
from skimage import img_as_float
from skimage.restoration import estimate_sigma


class NoiseReduction:

    def __init__(self, sigma_threshold=0.02):
        self.sigma_threshold = sigma_threshold

    def detect(self, image):
        """
        Estimate noise level using skimage estimate_sigma
        """
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_float = img_as_float(img_rgb)

        sigma_est = float(
            estimate_sigma(img_float, channel_axis=-1, average_sigmas=True)
        )

        needed = sigma_est > self.sigma_threshold

        return {
            "needed": needed,
            "estimated_sigma": sigma_est,
            "threshold": self.sigma_threshold
        }

    def apply(self, image):
        """
        Apply Non-Local Means Denoising
        """
        denoised = cv2.fastNlMeansDenoisingColored(
            image,
            None,
            h=5,
            hColor=10,
            templateWindowSize=7,
            searchWindowSize=21
        )

        return denoised