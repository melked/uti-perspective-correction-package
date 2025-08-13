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
# Yardımcı Fonksiyonlar

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
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(round(max(widthA, widthB)))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(round(max(heightA, heightB)))
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0,maxHeight-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LANCZOS4)


def _full_image_quad(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    return np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=np.float32)


def _unsharp_mask(image, ksize=(5,5), strength=1.5):
    blur = cv2.GaussianBlur(image, ksize, 0)
    return cv2.addWeighted(image, 1+strength, blur, -strength, 0)


def _gamma_correction(image: np.ndarray, gamma=1.5) -> np.ndarray:
    invGamma = 1.0 / gamma
    table = np.array([(i/255.0)**invGamma*255 for i in range(256)]).astype('uint8')
    return cv2.LUT(image, table)

# ------------------------------
# Hough Lines & Intersection

def _line_intersection(l1, l2):
    x1, y1, x2, y2 = map(np.float64, l1)
    x3, y3, x4, y4 = map(np.float64, l2)
    denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
    if np.abs(denom) < 1e-6:
        return None
    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
    return [px, py]

# ------------------------------
# Adaptive Preprocessing

def _preprocess_black_top_hat(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Black hat (karanlık çizgiler öne çıkar)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    # Top hat (aydınlık çizgiler öne çıkar)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    combined = cv2.addWeighted(blackhat, 0.5, tophat, 0.5, 0)
    # Bilateral filter + Unsharp
    filtered = cv2.bilateralFilter(combined, 9, 75, 75)
    sharpened = _unsharp_mask(filtered)
    # Normalize
    return cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# ------------------------------
# Detect Document using Hough + Intersections

def _detect_document_quad(image: np.ndarray) -> np.ndarray:
    preprocessed = _preprocess_black_top_hat(image)
    edges = cv2.Canny(preprocessed, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=50, maxLineGap=10)
    if lines is None or len(lines) < 4:
        return _full_image_quad(image)

    # Toleranslı açı filtrelemesi
    filtered = []
    for line in lines:
        x1,y1,x2,y2 = line[0]
        angle = np.degrees(np.arctan2(y2-y1, x2-x1)) % 180
        if abs(angle-0)<15 or abs(angle-90)<15 or abs(angle-180)<15:
            filtered.append(line)
    lines = np.array(filtered) if filtered else None

    if lines is None or len(lines)<4:
        return _full_image_quad(image)

    # Kesişim noktalarını bul
    points = []
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            inter = _line_intersection(lines[i][0], lines[j][0])
            if inter is not None:
                points.append(inter)
    if len(points) < 4:
        return _full_image_quad(image)

    points = np.array(points, dtype=np.float32)
    hull = cv2.convexHull(points)
    if len(hull) < 4:
        return _full_image_quad(image)
    return hull[:4].reshape(4,2)

# ------------------------------
# Executor Sınıfı

class PerspectiveCorrection(Component):
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
        if img is None or img.size==0:
            raise ValueError("Input image is empty or None.")
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if img.ndim==2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[-1]==4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def run(self):
        img_obj = Image.get_frame(img=self.image, redis_db=self.redis_db)
        if img_obj is None or img_obj.value is None:
            raise ValueError("No input image provided or failed to load.")

        src_img = self._prepare_image(img_obj.value)
        best_quad = _detect_document_quad(src_img)
        warped = _four_point_transform(src_img, best_quad)

        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)

        self.context["src_quad"] = best_quad.tolist()
        self.context["output_size"] = [warped.shape[1], warped.shape[0]]

        return build_response(context=self)

# ------------------------------
if __name__ == "__main__":
    Executor(sys.argv[1]).run()
