import cv2
import numpy as np


class IlluminationCorrection:

    def __init__(self, clip_limit=5.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def detect(self, image):
        """
        Decide whether illumination correction is needed.
        Simple logic: check brightness contrast spread.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        std_dev = float(np.std(gray))

        # Low contrast = low std deviation
        needed = std_dev < 40   # you can tune this

        return {
            "needed": needed,
            "contrast_std": std_dev,
            "threshold": 40
        }

    def apply(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size
        )

        l_clahe = clahe.apply(l)
        enhanced_lab = cv2.merge((l_clahe, a, b))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        return enhanced