import os
import sys
import cv2
import numpy as np
from typing import List

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveCorrection.src.utils.response import build_response
from components.PerspectiveCorrection.src.models.PackageModel import PackageModel


def order_points(pts):
    """Order points: top-left, top-right, bottom-right, bottom-left"""
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect


def four_point_transform(image, pts):
    """Apply perspective transform using 4 corner points"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    dst = np.array([
        [0, 0], [max_width - 1, 0],
        [max_width - 1, max_height - 1], [0, max_height - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (max_width, max_height))


def detect_document(image):
    """Detect document contour and return corner points"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    edged = cv2.Canny(blurred, 50, 200)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours[:5]:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)

    return None


def full_image_quad(image):
    """Return full image corners as fallback"""
    h, w = image.shape[:2]
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)


class PerspectiveCorrection:
    def __init__(self, request, bootstrap):
        self.request = request
        self.bootstrap = bootstrap
        self.context = {}
        self.image = self.request.get_param("inputImage")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def run(self):
        # Get image from request
        img_obj = self.request.get_image(self.image)
        if img_obj is None or img_obj.value is None:
            raise ValueError("No input image provided.")

        src = img_obj.value

        # Ensure proper image format
        if src.dtype != np.uint8:
            src = cv2.normalize(src, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if src.ndim == 2:
            src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
        elif src.shape[-1] == 4:
            src = cv2.cvtColor(src, cv2.COLOR_BGRA2BGR)

        # Detect document corners
        corners = detect_document(src)
        if corners is None:
            corners = full_image_quad(src)

        # Apply perspective correction
        warped = four_point_transform(src, corners)

        # Update image object
        img_obj.value = warped
        self.image = self.request.set_image(img_obj)

        # Set context
        self.context["src_quad"] = corners.tolist()
        self.context["output_size"] = [warped.shape[1], warped.shape[0]]

        return {"success": True, "context": self.context}


if __name__ == "__main__":
    Executor(sys.argv[1]).run()
