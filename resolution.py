import cv2
import numpy as np


class ResolutionEnhancement:

    def __init__(self, text_height_threshold=30, scale_factor=2.0):
        self.text_height_threshold = text_height_threshold
        self.scale_factor = scale_factor

    def detect(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        mser = cv2.MSER_create()
        mser.setMinArea(50)
        mser.setMaxArea(5000)

        regions, _ = mser.detectRegions(gray)

        text_heights = []

        for region in regions:
            x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))

            if w * h > image.shape[0] * image.shape[1] * 0.2:
                continue

            text_heights.append(h)

        if text_heights:
            avg_text_height = float(np.mean(text_heights))
        else:
            avg_text_height = float(image.shape[0])

        needed = avg_text_height < self.text_height_threshold

        return {
            "needed": needed,
            "avg_text_height": avg_text_height,
            "threshold": self.text_height_threshold,
            "scale_factor": self.scale_factor
        }

    def apply(self, image):

        new_width = int(image.shape[1] * self.scale_factor)
        new_height = int(image.shape[0] * self.scale_factor)

        upscaled = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_CUBIC
        )

        return upscaled