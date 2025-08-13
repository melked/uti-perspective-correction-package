import os
import sys
import cv2
import numpy as np
from typing import List, Tuple
from itertools import combinations

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveCorrection.src.utils.response import build_response
from components.PerspectiveCorrection.src.models.PackageModel import PackageModel


# ------------------------------
# Yardımcı Fonksiyonlar
def _order_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left
    return rect

def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect
    maxWidth = int(round(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))))
    maxHeight = int(round(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))))
    maxWidth = max(1, maxWidth)
    maxHeight = max(1, maxHeight)
    dst = np.array([[0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LANCZOS4)

def _full_image_quad(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    return np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)

def _unsharp_mask(image, ksize=(5, 5), strength=1.5):
    blur = cv2.GaussianBlur(image, ksize, 0)
    return cv2.addWeighted(image, 1 + strength, blur, -strength, 0)

def _gamma_correction(image: np.ndarray, gamma=1.5) -> np.ndarray:
    invGamma = 1.0 / float(gamma)
    table = (np.linspace(0, 1, 256) ** invGamma * 255.0).astype("uint8")
    return cv2.LUT(image, table)


# ------------------------------
# Ön işleme yöntemleri (tek kanal 8-bit geri döner)
def _preprocess_clahe(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def _preprocess_gamma(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = float(np.mean(gray))
    gamma = 1.8 if mean < 80 else (0.6 if mean > 180 else 1.0)
    corr = _gamma_correction(gray, gamma)
    return corr

def _preprocess_black_top_hat(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    combined = cv2.addWeighted(blackhat, 1.0, tophat, 1.0, 0)
    return cv2.GaussianBlur(combined, (5, 5), 0)

def _preprocess_color_segmentation(image: np.ndarray) -> np.ndarray:
    # Açık renkli/kağıt benzeri alanları maskele
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 60, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


# ------------------------------
# Geometri kontrolleri ve skor
def _is_valid_quad(quad: np.ndarray, w: int, h: int, min_area_ratio: float = 0.05) -> bool:
    q = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    # sınır içi
    if not np.all((q[:, 0] >= -1) & (q[:, 0] <= w) & (q[:, 1] >= -1) & (q[:, 1] <= h)):
        return False
    # alan ve konvekslik
    area = cv2.contourArea(q)
    if area < min_area_ratio * (w * h):
        return False
    if cv2.isContourConvex(q) is False:
        return False
    return True

def _score_quad(quad: np.ndarray, w: int, h: int) -> float:
    # alan * dikdörtgensellik (alan / boundRect alanı)
    area = float(cv2.contourArea(quad.astype(np.float32)))
    x, y, bw, bh = cv2.boundingRect(quad.astype(np.float32))
    rect_area = max(1.0, float(bw * bh))
    rectangularity = area / rect_area
    return area * rectangularity


# ------------------------------
# Hough & kesişim tabanlı köşe tespiti (overflow-safe)
def _filter_lines(lines, angle_tol=15):
    if lines is None:
        return None
    filtered = []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        angle = (np.degrees(np.arctan2(y2 - y1, x2 - x1)) + 180.0) % 180.0
        if min(abs(angle - 0), abs(angle - 90), abs(angle - 180)) < angle_tol:
            filtered.append(l)
    return np.array(filtered) if filtered else None

def _line_intersections(lines: np.ndarray, w: int, h: int) -> np.ndarray:
    pts = []
    if lines is None or len(lines) < 2:
        return np.empty((0, 2), dtype=np.float32)

    # float64'e çevir – overflow fix
    L = lines[:, 0, :].astype(np.float64)
    for (x1, y1, x2, y2), (x3, y3, x4, y4) in combinations(L, 2):
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:
            continue
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
        if np.isfinite(px) and np.isfinite(py):
            # görüntü sınırına kırp
            px = float(np.clip(px, 0, w - 1))
            py = float(np.clip(py, 0, h - 1))
            pts.append([px, py])
    return np.array(pts, dtype=np.float32)

def _compute_corners_from_lines(image: np.ndarray, edges: np.ndarray) -> np.ndarray:
    h, w = edges.shape[:2]
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=min(w, h) // 6, maxLineGap=10)
    lines = _filter_lines(lines, 15)
    if lines is None or len(lines) < 4:
        return _full_image_quad(image)

    points = _line_intersections(lines, w, h)
    if points.shape[0] < 4:
        return _full_image_quad(image)

    hull = cv2.convexHull(points.astype(np.float32))
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
    if approx.shape[0] != 4:
        return _full_image_quad(image)
    quad = approx.reshape(4, 2).astype(np.float32)
    return quad

def _detect_by_contours(image: np.ndarray, pre: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    # kenarları kapatmak için hafif genleşme + erozyon
    edges = cv2.Canny(pre, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    for c in cnts[:10]:  # en büyük 10 konturu dene
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float32)
            if _is_valid_quad(quad, w, h):
                return quad
    return _full_image_quad(image)


# ------------------------------
# Ensemble ile en iyi dörtgen seçimi
def _detect_document(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    preprocessors = [
        _preprocess_clahe,
        _preprocess_gamma,
        _preprocess_black_top_hat,
        _preprocess_color_segmentation,
    ]

    candidates: List[np.ndarray] = []

    for pre_fn in preprocessors:
        pre = pre_fn(image)
        # Canny
        edges = cv2.Canny(pre, 50, 150)
        # 1) Hough+kesişim
        quad_hough = _compute_corners_from_lines(image, edges)
        if _is_valid_quad(quad_hough, w, h):
            candidates.append(quad_hough)
        # 2) Kontur tabanlı
        quad_cnt = _detect_by_contours(image, pre)
        if _is_valid_quad(quad_cnt, w, h):
            candidates.append(quad_cnt)

    if not candidates:
        return _full_image_quad(image)

    # en iyi skoru seç
    scores = [(_score_quad(q, w, h), q) for q in candidates]
    best_quad = max(scores, key=lambda t: t[0])[1]
    return best_quad.astype(np.float32)

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

        best_quad = _detect_document(src_img)
        warped = _four_point_transform(src_img, best_quad)

        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)

        self.context["src_quad"] = np.asarray(best_quad, dtype=float).tolist()
        self.context["output_size"] = [int(warped.shape[1]), int(warped.shape[0])]
        return build_response(context=self)


if __name__ == "__main__":
    Executor(sys.argv[1]).run()
