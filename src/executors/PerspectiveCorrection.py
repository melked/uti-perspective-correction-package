import os
import sys
import cv2
import numpy as np
import math
from typing import Optional, List, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveCorrection.src.utils.response import build_response
from components.PerspectiveCorrection.src.models.PackageModel import PackageModel


# -----------------------------------------------------------------------------
# 1. PARAMETRE YÖNETİMİ SINIFI
# -----------------------------------------------------------------------------
class Params:
    """Tüm algoritma parametrelerini yönetmek için merkezi bir sınıf."""

    def __init__(self, config=None):
        config = config or {}
        self.resize_longest_edge = config.get("resize_longest_edge", 1000)
        self.unsharp_strength = config.get("unsharp_strength", 1.5)
        self.score_min_area_ratio = config.get("score_min_area_ratio", 0.10)
        self.score_max_area_ratio = config.get("score_max_area_ratio", 0.98)
        self.approx_poly_epsilon_ratio = config.get("approx_poly_epsilon_ratio", 0.02)
        self.min_confidence_threshold = config.get("min_confidence_threshold", 0.25)
        self.hough_line_threshold = config.get("hough_line_threshold", 50)
        self.hough_min_line_length = config.get("hough_min_line_length", 50)
        self.hough_max_line_gap = config.get("hough_max_line_gap", 20)
        self.hough_angle_tolerance = config.get("hough_angle_tolerance", 10)


# -----------------------------------------------------------------------------
# 2. YARDIMCI & GEOMETRİ FONKSİYONLARI
# -----------------------------------------------------------------------------
def _order_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1);
    rect[0] = pts[np.argmin(s)];
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1);
    rect[1] = pts[np.argmin(diff)];
    rect[3] = pts[np.argmax(diff)]
    return rect


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> Optional[np.ndarray]:
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl);
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br);
    heightB = np.linalg.norm(tl - bl)
    maxWidth = max(int(widthA), int(widthB));
    maxHeight = max(int(heightA), int(heightB))
    if maxWidth <= 10 or maxHeight <= 10: return None
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LANCZOS4)


def _unsharp_mask(image: np.ndarray, strength: float) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    return cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)


def _line_intersection(line1, line2) -> Optional[Tuple[int, int]]:
    """İki çizginin kesişim noktasını hesaplar."""
    x1, y1, x2, y2 = line1[0]
    x3, y3, x4, y4 = line2[0]
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0: return None  # Çizgiler paralel
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    if 0 <= t <= 1 and 0 <= u <= 1:  # Kesişim segmentler üzerinde mi kontrolü (opsiyonel ama iyi)
        ix = int(x1 + t * (x2 - x1))
        iy = int(y1 + t * (y2 - y1))
        return (ix, iy)
    return None


# -----------------------------------------------------------------------------
# 3. UZMAN STRATEJİLERİ (HOUGH GÜÇLENDİRİLDİ)
# -----------------------------------------------------------------------------
def _get_quads_from_contours(contours: List[np.ndarray], params: Params) -> List[np.ndarray]:
    quads = []
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, params.approx_poly_epsilon_ratio * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quads.append(approx.reshape(4, 2).astype(np.float32))
    return quads


def strategy_canny(image: np.ndarray, params: Params) -> List[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    v = np.median(blurred);
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v));
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(blurred, lower, upper)
    closed = cv2.morphologyEx(edged, cv2.MORPH_RECT, (7, 7), iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return _get_quads_from_contours(contours, params)


def strategy_hough_lines(image: np.ndarray, params: Params) -> List[np.ndarray]:
    """Düz çizgileri bularak ve kesiştirerek dörtgen adayları üretir."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edged, 1, np.pi / 180,
                            threshold=params.hough_line_threshold,
                            minLineLength=params.hough_min_line_length,
                            maxLineGap=params.hough_max_line_gap)
    if lines is None: return []

    horizontal, vertical = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        if angle < params.hough_angle_tolerance or abs(angle - 180) < params.hough_angle_tolerance:
            horizontal.append(line)
        elif abs(angle - 90) < params.hough_angle_tolerance:
            vertical.append(line)

    if len(horizontal) < 2 or len(vertical) < 2: return []

    # En dıştaki çizgileri kenar olarak seç
    horizontal.sort(key=lambda line: line[0][1])
    vertical.sort(key=lambda line: line[0][0])
    top_line, bottom_line = horizontal[0], horizontal[-1]
    left_line, right_line = vertical[0], vertical[-1]

    # Kenarların kesişim noktalarını hesapla
    tl = _line_intersection(top_line, left_line)
    tr = _line_intersection(top_line, right_line)
    bl = _line_intersection(bottom_line, left_line)
    br = _line_intersection(bottom_line, right_line)

    if all((tl, tr, bl, br)):
        quad = np.array([tl, tr, br, bl], dtype=np.float32)
        return [quad]
    return []


# -----------------------------------------------------------------------------

def _score_quad(quad: np.ndarray, params: Params, image_shape: tuple) -> float:
    """Doğrudan bir dörtgeni puanlar."""
    h, w = image_shape[:2];
    total_area = w * h
    contour = quad.astype(np.int32)
    area = cv2.contourArea(contour)
    if not (params.score_min_area_ratio < area / total_area < params.score_max_area_ratio): return 0.0
    M = cv2.moments(contour);
    if M["m00"] == 0: return 0.0
    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    centrality = 1.0 - (np.linalg.norm(np.array([cx, cy]) - np.array([w / 2, h / 2])) / (max(w, h) / 2))
    (tl, tr, br, bl) = _order_points(quad)
    width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    height = (np.linalg.norm(tl - bl) + np.linalg.norm(tr - br)) / 2
    if min(width, height) < 1: return 0.0
    aspect_ratio = max(width, height) / min(width, height)
    aspect_score = 1.0 if 1.1 < aspect_ratio < 2.2 else 0.5
    return (area / total_area * 0.4) + (centrality * 0.4) + (aspect_score * 0.2)


# -----------------------------------------------------------------------------
# 5. ANA BİLEŞEN
# -----------------------------------------------------------------------------
class PerspectiveCorrection(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.context = {}
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")
        params_data = self.request.get_param("params", {})
        self.params = Params(params_data)

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def _prepare_image(self, img: np.ndarray) -> np.ndarray:
        if img is None or img.size == 0: raise ValueError("Input image is empty or None.")
        if img.dtype != np.uint8: img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def run(self):
        img_obj = Image.get_frame(img=self.image, redis_db=self.redis_db)
        if img_obj is None or img_obj.value is None: raise ValueError("No input image provided or failed to load.")
        src_img_orig = self._prepare_image(img_obj.value)
        h, w = src_img_orig.shape[:2]
        scale = self.params.resize_longest_edge / max(h, w) if max(h, w) > self.params.resize_longest_edge else 1
        work_img = cv2.resize(src_img_orig, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        work_img = _unsharp_mask(work_img, self.params.unsharp_strength)
        document_quad = None

        print("Tüm uzman stratejileri çalıştırılıyor...")
        quad_candidates = []
        quad_candidates.extend(strategy_hough_lines(work_img, self.params))
        quad_candidates.extend(strategy_canny(work_img, self.params))

        if not quad_candidates:
            print("Hiçbir strateji aday üretemedi.")
            warped = None
        else:
            print(f"{len(quad_candidates)} aday dörtgen bulundu. En iyisi seçiliyor...")
            scored_candidates = [(_score_quad(q, self.params, work_img.shape), q) for q in quad_candidates]
            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            best_score, best_quad = scored_candidates[0]
            if best_score > self.params.min_confidence_threshold:
                print(f"En iyi aday {best_score:.2f} puanla bulundu.")
                document_quad = best_quad
            else:
                print(
                    f"En iyi adayın puanı ({best_score:.2f}) minimum eşiğin ({self.params.min_confidence_threshold}) altında kaldı.")
                document_quad = None

        warped = None
        if document_quad is not None:
            document_quad /= scale
            warped = _four_point_transform(src_img_orig, document_quad)

        if warped is None:
            print("Tüm analizler başarısız. Fallback olarak orijinal görüntü kullanılıyor.")
            warped = src_img_orig
            document_quad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)
        self.context["src_quad"] = document_quad.tolist()
        self.context["output_size"] = [warped.shape[1], warped.shape[0]]
        return build_response(context=self)


# -----------------------------------------------------------------------------
# 6. ÇALIŞTIRICI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    Executor(sys.argv[1]).run()