import cv2 
import os
import numpy as np
from .base import BaseTechnique
from utils.image_utils import order_points,four_point_transform,save_output

class PerspectiveCorrection(BaseTechnique):

    def __init__(self, output_dir="output"):
        super().__init__("perspective_correction", output_dir)

    def detect(self, image, metadata):
        corners = metadata.get("corners")

        if corners is None:
            return {
                "needed": False,
                "confidence": 0.0,
                "metrics": {},
                "reason": "No corners available"
            }

        rect = order_points(corners)

        def angle(ptA, ptB, ptC):
            BA = ptA - ptB
            BC = ptC - ptB
            cos_angle = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC))
            return np.degrees(np.arccos(cos_angle))

        angles = [angle(rect[i-1], rect[i], rect[(i+1)%4]) for i in range(4)]
        deviation = np.mean([abs(a - 90) for a in angles])

        return {
            "needed": deviation > 5,
            "confidence": float(min(deviation / 45, 1.0)),
            "metrics": {"angle_deviation": float(deviation)},
            "reason": "Perspective distortion detected" if deviation > 5 else "Already aligned"
        }

    def apply(self, image, metadata):
        corners = metadata["corners"]
        warped = four_point_transform(image, corners)

        #path = save_output(warped, self.output_dir, self.name, "applied")

        return {
            "processed_image": warped,
            "output_path": path,
            "metrics_after": {}
        }