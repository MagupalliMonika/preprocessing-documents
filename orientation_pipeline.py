import cv2
import numpy as np
import pytesseract
from pytesseract import TesseractError


# ==========================================================
# STEP 1: 90° Orientation Detection
# ==========================================================
def detect_and_correct_orientation(image, confidence_threshold=1.5):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    try:
        osd = pytesseract.image_to_osd(
            thresh,
            config="--psm 0",
            output_type=pytesseract.Output.DICT
        )

        angle = osd["rotate"]
        confidence = float(osd["orientation_conf"])

    except TesseractError:
        return image, 0, 0.0, False

    if confidence >= confidence_threshold and angle in [90, 180, 270]:

        if angle == 90:
            corrected = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif angle == 180:
            corrected = cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            corrected = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        return corrected, angle, confidence, True

    return image, 0, confidence, False


# ==========================================================
# STEP 2: Small Skew Detection
# ==========================================================
def detect_and_correct_skew(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    coords = np.column_stack(np.where(thresh > 0))

    if len(coords) < 100:
        return image, 0.0, False

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]

    if angle > 45:
        angle = angle - 90
    elif angle < -45:
        angle = 90 + angle

    if abs(angle) < 0.5:
        return image, 0.0, False

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, -angle, 1.0)

    rotated = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated, float(angle), True


# ==========================================================
# FINAL ORIENTATION PIPELINE
# ==========================================================
def correct_document_orientation(image):

    rotated_img, rot_angle, confidence, applied_90 = \
        detect_and_correct_orientation(image)

    if applied_90:
        final = rotated_img
        skew_angle = 0.0
        applied_skew = False
    else:
        final, skew_angle, applied_skew = \
            detect_and_correct_skew(rotated_img)

    results = {
        "orientation_correction": {
            "rotation_angle_90": rot_angle,
            "rotation_confidence": confidence,
            "rotation_applied": applied_90,
            "skew_angle": skew_angle,
            "skew_applied": applied_skew
        }
    }

    return final, results