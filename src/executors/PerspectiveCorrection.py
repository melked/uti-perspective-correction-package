import os
import sys
import cv2
import numpy as np
from typing import List

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveCorrection.src.utils.response import build_response
from components.PerspectiveCorrection.src.models.PackageModel import PackageModel

# ------------------------------
# Yardımcı fonksiyonlar
def _order_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect
    maxWidth = int(round(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))))
    maxHeight = int(round(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))))
    dst = np.array([[0, 0], [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LANCZOS4)

def _full_image_quad(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

def _unsharp_mask(image, ksize=(5, 5), strength=1.5):
    blur = cv2.GaussianBlur(image, ksize, 0)
    return cv2.addWeighted(image, 1 + strength, blur, -strength, 0)

def _gamma_correction(image, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

# ------------------------------
# Ön işleme (optimize parametrelerle)
def _preprocess(image: np.ndarray, method: str) -> np.ndarray:
    img = image.copy()

    if method == "clahe":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(gray)

    elif method == "sharpen_bilateral":
        img = cv2.bilateralFilter(img, 9, 75, 75)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        clahe_L = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(L)
        img = cv2.cvtColor(cv2.merge((clahe_L, A, B)), cv2.COLOR_LAB2BGR)
        img = _unsharp_mask(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    elif method == "gamma":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gamma_val = 1.8 if np.mean(gray) < 80 else (0.6 if np.mean(gray) > 180 else 1.0)
        img = _gamma_correction(img, gamma_val)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    elif method == "dark":
        L, _, _ = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
        _, img = cv2.threshold(L, 80, 255, cv2.THRESH_BINARY_INV)

    elif method == "light":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 30, 255]))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # Kenar tespiti
    edges = cv2.Canny(img, 40, 160)  # optimize edilmiş threshold
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))  # daha büyük kernel
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# ------------------------------
# HoughLines ile köşe bulma
def _find_quad(binary_img: np.ndarray, ref_image: np.ndarray) -> np.ndarray:
    # Önce kontur denemesi
    contours, _ = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                return approx.reshape(4, 2).astype(np.float32)

    # HoughLinesP ile çizgi bulma
    lines = cv2.HoughLinesP(binary_img, 1, np.pi / 180, threshold=120,
                            minLineLength=100, maxLineGap=15)
    if lines is None:
        return _full_image_quad(ref_image)

    intersections = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            p1 = lines[i][0]
            p2 = lines[j][0]
            inter = _line_intersection(p1, p2)
            if inter is not None:
                intersections.append(inter)

    if len(intersections) >= 4:
        pts = np.array(intersections, dtype=np.float32)
        rect = cv2.convexHull(pts)
        if len(rect) >= 4:
            return rect[:4].reshape(4, 2)

    return _full_image_quad(ref_image)

def _line_intersection(l1, l2):
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None
    px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / denom
    return [px, py]

# ------------------------------
# Ana belge tespiti
def detect_document(image: np.ndarray) -> np.ndarray:
    methods = ["clahe", "sharpen_bilateral", "gamma", "dark", "light"]
    candidates = []
    for m in methods:
        edges = _preprocess(image, m)
        quad = _find_quad(edges, image)
        if not any(np.allclose(quad, c, atol=2) for c in candidates):
            candidates.append(quad)

    if not candidates:
        candidates.append(_full_image_quad(image))

    return max(candidates, key=lambda q: cv2.contourArea(q.astype(np.float32)))

# ------------------------------
# Executor sınıfı
class PerspectiveTransformation(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.context = {}
        data_dict = getattr(self.request, "data", {}) or {}
        self.request.model = PackageModel(**data_dict)
        self.image = self.request.get_param("inputImage")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def _prepare_image(self, img: np.ndarray) -> np.ndarray:
        if img is None or img.size == 0:
            raise ValueError("Input image is empty or None.")
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def run(self):
        img_obj = Image.get_frame(img=self.image, redis_db=self.redis_db)
        if img_obj is None or img_obj.value is None:
            raise ValueError("No input image provided or failed to load.")

        src_img = self._prepare_image(img_obj.value)
        best_quad = detect_document(src_img)
        warped = _four_point_transform(src_img, best_quad)

        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)

        self.context["src_quad"] = best_quad.tolist()
        self.context["output_size"] = [warped.shape[1], warped.shape[0]]

        return build_response(context=self)

# ------------------------------
if __name__ == "__main__":
    Executor(sys.argv[1]).run()
