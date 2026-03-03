import cv2
import numpy as np
from .base import BaseTechnique
from utils.image_utils import save_output  

class DocumentBoundaryDetector(BaseTechnique):

    def __init__(self, mask_generator, output_dir="output"):
        super().__init__("document_boundary", output_dir)
        self.mask_generator = mask_generator  # FIXED

    def detect(self, image, metadata=None):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = self.mask_generator.generate(rgb)

        if not masks:
            return {
                "needed": False,
                "confidence": 0.0,
                "metrics": {},
                "reason": "No masks detected",
                "corners": None
            }

        largest_mask = max(masks, key=lambda x: x['area'])
        mask = largest_mask['segmentation'].astype("uint8") * 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return {
                "needed": False,
                "confidence": 0.0,
                "metrics": {},
                "reason": "No contour found",
                "corners": None
            }

        c = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        area_ratio = cv2.contourArea(c) / (image.shape[0] * image.shape[1])

        return {
            "needed": len(approx) == 4,
            "confidence": float(area_ratio),
            "metrics": {"area_ratio": float(area_ratio)},
            "reason": "Quadrilateral detected" if len(approx) == 4 else "Not quadrilateral",
            "corners": approx.reshape(4, 2) if len(approx) == 4 else None
        }

    def apply(self, image, metadata):

      # Save visualization instead of raw image
      preview = image.copy()

      if metadata["corners"] is not None:
          pts = metadata["corners"]
          cv2.polylines(preview, [pts], True, (0,255,0), 3)

      path = save_output(preview, self.output_dir, self.name, "analyzed")

      return {
          "processed_image": image,   # original image unchanged
          "output_path": path,
          "metrics_after": metadata
      }