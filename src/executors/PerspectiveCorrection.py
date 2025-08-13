import os
import sys
import cv2
import numpy as np
from typing import List, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveCorrection.src.utils.response import build_response
from components.PerspectiveCorrection.src.models.PackageModel import PackageModel

def _unsharp_mask(image, ksize=(5,5), strength=1.5):
    blur = cv2.GaussianBlur(image, ksize, 0)
    return cv2.addWeighted(image, 1 + strength, blur, -strength, 0)

def _gamma_correction(image: np.ndarray, gamma=1.5) -> np.ndarray:
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

def _adaptive_contrast_enhancement(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(L)
    lab_clahe = cv2.merge((cl,A,B))
    img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    return _gamma_correction(img_clahe, gamma=1.0)

def _prepare_image(img: np.ndarray) -> np.ndarray:
    if img is None or img.size == 0:
        raise ValueError("Input image is empty or None.")
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4,2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left
    return rect

def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br-bl)
    widthB = np.linalg.norm(tr-tl)
    maxWidth = int(round(max(widthA,widthB)))
    heightA = np.linalg.norm(tr-br)
    heightB = np.linalg.norm(tl-bl)
    maxHeight = int(round(max(heightA,heightB)))
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0,maxHeight-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth,maxHeight), flags=cv2.INTER_LANCZOS4)
    return warped

def _line_intersections(lines: np.ndarray) -> List[Tuple[float,float]]:
    """İki çizgi arasındaki kesişim noktalarını hesapla."""
    points = []
    if lines is None:
        return points
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            x1,y1,x2,y2 = lines[i][0]
            x3,y3,x4,y4 = lines[j][0]
            denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
            if denom == 0:
                continue
            px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4))/denom
            py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4))/denom
            points.append((px,py))
    return points

def _filter_corners(points: List[Tuple[float,float]], img_shape, tol=20) -> np.ndarray:
    """Çok yakın noktaları birleştir, maksimum 4 köşe seç."""
    filtered = []
    for p in points:
        if not any(np.linalg.norm(np.array(p)-np.array(f))<tol for f in filtered):
            filtered.append(p)
    # Resmin köşelerine en yakın noktaları al
    if len(filtered) > 4:
        h, w = img_shape[:2]
        filtered = sorted(filtered, key=lambda x: (x[0]-w/2)**2 + (x[1]-h/2)**2)
        filtered = filtered[:4]
    if len(filtered) < 4:
        # Eksikse tüm resmi kullan
        filtered = [(0,0),(w-1,0),(w-1,h-1),(0,h-1)]
    return np.array(filtered, dtype=np.float32)

def detect_corners_hough(img: np.ndarray) -> np.ndarray:
    """Ön işleme + HoughLines tabanlı köşe bulma."""
    gray = cv2.cvtColor(_adaptive_contrast_enhancement(img), cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=50, maxLineGap=10)
    points = _line_intersections(lines)
    corners = _filter_corners(points, img.shape, tol=20)
    return corners

# ------------------------------
# Executor Sınıfı
# ------------------------------
class PerspectiveTransformation(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.context = {}
        if not hasattr(self.request, "data") or self.request.data is None:
            raise ValueError("Request does not contain 'data'")
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def run(self):
        img_obj = Image.get_frame(img=self.image, redis_db=self.redis_db)
        if img_obj is None or img_obj.value is None:
            raise ValueError("No input image provided or failed to load.")
        src_img = _prepare_image(img_obj.value)

        # Köşe tespiti
        corners = detect_corners_hough(src_img)

        # Perspektif dönüşüm uygula
        warped = _four_point_transform(src_img, corners)

        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)

        self.context["src_quad"] = corners.tolist()
        self.context["output_size"] = [warped.shape[1], warped.shape[0]]

        return build_response(context=self)

if __name__ == "__main__":
    Executor(sys.argv[1]).run()
