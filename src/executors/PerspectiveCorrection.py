import os
import sys
import cv2
import numpy as np
from typing import List, Tuple, Optional
from itertools import combinations

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveCorrection.src.utils.response import build_response
from components.PerspectiveCorrection.src.models.PackageModel import PackageModel


# ---------------------- Geometry Helpers ---------------------- #

def _order_points(pts: np.ndarray) -> np.ndarray:
    """Köşeleri tl, tr, br, bl sırasına sokar."""
    pts = pts.reshape(4, 2).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
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
    maxWidth, maxHeight = max(2, maxWidth), max(2, maxHeight)
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LANCZOS4)


def _full_image_quad(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)


def _min_area_rect_quad(binary_or_gray: np.ndarray, ref_image: np.ndarray) -> np.ndarray:
    gray = binary_or_gray if len(binary_or_gray.shape) == 2 else cv2.cvtColor(binary_or_gray, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return _full_image_quad(ref_image)
    c = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    return cv2.boxPoints(rect).astype(np.float32)


# ---------------------- Preprocessing Pipelines ---------------------- #

def _pre_soft(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 11)
    th = cv2.medianBlur(th, 5)
    return cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)


def _pre_medium(gray: np.ndarray) -> np.ndarray:
    eq = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    edges = cv2.Canny(eq, 50, 150, L2gradient=True)
    return cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)


def _pre_hard(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    unsharp = cv2.addWeighted(gray, 1.7, blur, -0.7, 0)
    edges = cv2.Canny(unsharp, 30, 100, L2gradient=True)
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)


def _create_background_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Doygun (renkli) ve/veya karanlık arka plan alanlarını tespit eder."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    # Doygunluk (saturation) eşiği ile renkli alanları bul (ahşap, kırmızı vb.)
    _, sat_mask = cv2.threshold(saturation, 40, 255, cv2.THRESH_BINARY)
    # Değer (parlaklık) eşiği ile çok koyu alanları bul
    _, val_mask = cv2.threshold(value, 40, 255, cv2.THRESH_BINARY_INV)
    # İki maskeyi birleştirerek hem renkli hem de koyu arka planları hedefle
    mask = cv2.bitwise_or(sat_mask, val_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)


def _pre_color_robust(image_bgr: np.ndarray) -> np.ndarray:
    """Renk maskeleme ve LAB renk uzayını kullanarak zorlu arka planları eler."""
    background_mask = _create_background_mask(image_bgr)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)
    l_enhanced[background_mask == 255] = 255
    blur = cv2.GaussianBlur(l_enhanced, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150, L2gradient=True)
    return cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)


def _generate_edge_maps(image_bgr: np.ndarray) -> List[np.ndarray]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return [
        _pre_soft(gray),
        _pre_medium(gray),
        _pre_hard(gray),
        _pre_color_robust(image_bgr)
    ]


# ---------------------- Candidate Generation ---------------------- #

def _find_quads_from_contours(binary_img: np.ndarray, ref_image: np.ndarray) -> List[np.ndarray]:
    contours, _ = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return []
    H, W = ref_image.shape[:2]
    min_area = H * W * 0.10
    candidates = []
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        if cv2.contourArea(c) < min_area: continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            candidates.append(approx.reshape(4, 2).astype(np.float32))
    return candidates


def _find_quads_from_hough(edges: np.ndarray, ref_image: np.ndarray) -> List[np.ndarray]:
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=10)
    if lines is None: return []

    H, W = ref_image.shape[:2]
    horizontal_lines, vertical_lines = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle < 30 or angle > 150:
            horizontal_lines.append(line[0])
        elif 60 < angle < 120:
            vertical_lines.append(line[0])

    if len(horizontal_lines) < 2 or len(vertical_lines) < 2: return []

    def line_intersection(line1, line2):
        x1, y1, x2, y2 = line1;
        x3, y3, x4, y4 = line2
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if den == 0: return None
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den
        if 0 < t < 1 and u > 0:
            return [int(x1 + t * (x2 - x1)), int(y1 + t * (y2 - y1))]
        return None

    quads = []
    for h1, h2 in combinations(horizontal_lines, 2):
        for v1, v2 in combinations(vertical_lines, 2):
            p1 = line_intersection(h1, v1)
            p2 = line_intersection(h1, v2)
            p3 = line_intersection(h2, v1)
            p4 = line_intersection(h2, v2)
            if all(p is not None for p in [p1, p2, p3, p4]):
                quads.append(np.array([p1, p2, p4, p3], dtype=np.float32))
    return quads


# ---------------------- Scoring & Selection ---------------------- #

def _score_quad(quad: np.ndarray, image: np.ndarray, edges: np.ndarray) -> float:
    area = abs(cv2.contourArea(quad))
    img_area = image.shape[0] * image.shape[1]

    if not cv2.isContourConvex(quad) or area < 0.1 * img_area:
        return 0.0

    area_score = np.clip(area / (0.9 * img_area), 0, 1)

    q_ord = _order_points(quad)

    def angle(p1, p2, p3):
        v1, v2 = p1 - p2, p3 - p2;
        return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6), -1, 1)))

    angles = [angle(q_ord[i], q_ord[(i + 1) % 4], q_ord[(i + 2) % 4]) for i in range(4)]
    angle_score = np.mean([1 - abs(a - 90) / 45 for a in angles])

    def seg_angle(p1, p2): return np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0] + 1e-6)) % 180

    a1, a3 = seg_angle(q_ord[0], q_ord[1]), seg_angle(q_ord[3], q_ord[2])
    a2, a4 = seg_angle(q_ord[1], q_ord[2]), seg_angle(q_ord[0], q_ord[3])

    def closeness(a, b): d = min(abs(a - b), 180 - abs(a - b)); return 1 - d / 20  # Açı farkına daha duyarlı

    parallel_score = (closeness(a1, a3) + closeness(a2, a4)) / 2

    return (0.4 * area_score) + (0.3 * angle_score) + (0.3 * parallel_score)


def _select_best_quad(image: np.ndarray, edge_maps: List[np.ndarray]) -> np.ndarray:
    all_candidates = []
    for emap in edge_maps:
        all_candidates.extend(_find_quads_from_contours(emap, image))
        all_candidates.extend(_find_quads_from_hough(emap, image))

    if not all_candidates:
        return _min_area_rect_quad(edge_maps[0], image) if edge_maps else _full_image_quad(image)

    # Yinelenen adayları filtrele
    unique_candidates = []
    if all_candidates:
        all_candidates.sort(key=cv2.contourArea, reverse=True)
        unique_candidates.append(all_candidates[0])
        for c in all_candidates[1:]:
            if not any(np.linalg.norm(c.mean(axis=0) - uc.mean(axis=0)) < 30 for uc in unique_candidates):
                unique_candidates.append(c)

    scoring_edges = edge_maps[-1]  # En güvenilir (renk tabanlı) haritayı kullan
    scored_candidates = [(q, _score_quad(q, image, scoring_edges)) for q in unique_candidates]

    if not scored_candidates or max(scored_candidates, key=lambda x: x[1])[1] < 0.2:  # Minimum skor eşiği
        return _min_area_rect_quad(edge_maps[0], image) if edge_maps else _full_image_quad(image)

    return max(scored_candidates, key=lambda x: x[1])[0]


# ---------------------- Component ---------------------- #

class PerspectiveCorrection(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.context = {}
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    @staticmethod
    def _prepare_image(img: np.ndarray) -> np.ndarray:
        if img is None or img.size == 0: raise ValueError("Input image empty or None")
        if img.dtype != np.uint8: img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def run(self):
        src = self._prepare_image(Image.get_frame(img=self.image, redis_db=self.redis_db).value)
        edge_maps = _generate_edge_maps(src)
        best_quad = _select_best_quad(src, edge_maps)
        warped = _four_point_transform(src, best_quad)

        img_obj = Image.get_frame(img=self.image, redis_db=self.redis_db)
        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)
        self.context["src_quad"] = best_quad.tolist()
        self.context["output_size"] = [warped.shape[1], warped.shape[0]]
        self.context["method"] = {
            "pipelines": ["soft", "medium", "hard", "color_robust"],
            "fallbacks": ["hough_lines", "min_area_rect"],
        }
        return build_response(context=self)


Executor(sys.argv[1]).run()