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

# ==============================
# Yardımcı – Sayısal Güvenli Fonksiyonlar
# ==============================

def _safe_float64(arr):
    return np.asarray(arr, dtype=np.float64)

def _order_points(pts: np.ndarray) -> np.ndarray:
    """ Dört köşeyi (tl, tr, br, bl) sıralar. """
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
    """ Köşelerden hedefe perspektif dönüşümü. """
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect
    maxWidth  = int(round(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))))
    maxHeight = int(round(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))))
    maxWidth  = max(1, maxWidth)
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

def _unsharp_mask(image, ksize=(5, 5), strength=1.0):
    blur = cv2.GaussianBlur(image, ksize, 0)
    return cv2.addWeighted(image, 1 + float(strength), blur, -float(strength), 0)

def _gamma_correction(image: np.ndarray, gamma=1.5) -> np.ndarray:
    invGamma = 1.0 / float(gamma)
    table = (np.linspace(0, 1, 256) ** invGamma * 255.0).astype("uint8")
    return cv2.LUT(image, table)

def _estimate_brightness(gray: np.ndarray) -> float:
    return float(np.mean(gray))

def _estimate_blur(gray: np.ndarray) -> float:
    # Laplacian variance: düşük değer = bulanık
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def _adaptive_canny_params(h: int, w: int, brightness: float, blur_var: float) -> Tuple[int, int]:
    base_low, base_high = 50, 150
    scale = max(0.6, min(1.4, 1200.0 / max(h, w)))  # küçük resimde threshold düşsün
    low, high = int(base_low * scale), int(base_high * scale)
    # karanlıksa şiddetlendir
    if brightness < 90:
        low = max(10, int(low * 0.7))
        high = max(40, int(high * 0.7))
    # çok bulanıksa threshold’u düşür
    if blur_var < 80:
        low = max(5, int(low * 0.6))
        high = max(25, int(high * 0.6))
    return low, high

def _poly_angle_score(quad: np.ndarray) -> float:
    """ 90° yakınlığına göre skor (1.0 iyi, 0 kötü). """
    q = _order_points(quad)
    def angle(a,b,c):
        ba = a - b
        bc = c - b
        cosang = np.clip(np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-6), -1, 1)
        ang = np.degrees(np.arccos(cosang))
        return ang
    angles = [
        angle(q[3], q[0], q[1]),
        angle(q[0], q[1], q[2]),
        angle(q[1], q[2], q[3]),
        angle(q[2], q[3], q[0]),
    ]
    diffs = [abs(a-90.0) for a in angles]
    mean_diff = sum(diffs) / 4.0
    return float(max(0.0, 1.0 - (mean_diff / 45.0)))  # 45° sapma -> 0 skor

def _rectangularity_score(quad: np.ndarray) -> float:
    q = quad.astype(np.float32)
    area = float(cv2.contourArea(q))
    x,y,w,h = cv2.boundingRect(q)
    rect_area = max(1.0, float(w*h))
    return float(area / rect_area)

def _size_ratio_ok(quad: np.ndarray, w: int, h: int, min_ratio=0.03, max_ratio=0.98) -> bool:
    area = float(cv2.contourArea(quad.astype(np.float32)))
    img_area = float(w*h)
    r = area / (img_area + 1e-6)
    return (r >= min_ratio) and (r <= max_ratio)

def _is_valid_quad(quad: np.ndarray, w: int, h: int) -> bool:
    q = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    # sınır içinde mi?
    if not np.all((q[:, 0] >= -2) & (q[:, 0] <= w+1) & (q[:, 1] >= -2) & (q[:, 1] <= h+1)):
        return False
    # alan, konvekslik, boyut oranı
    if cv2.contourArea(q) < 1.0:
        return False
    if cv2.isContourConvex(q) is False:
        return False
    if not _size_ratio_ok(q, w, h, min_ratio=0.04, max_ratio=0.995):
        return False
    return True

def _score_quad(quad: np.ndarray, w: int, h: int) -> float:
    """ Birleşik skor: alan * (rectangularity^1.5) * (angle_score^1.5) """
    area = float(cv2.contourArea(quad.astype(np.float32)))
    angle_s = _poly_angle_score(quad)
    rect_s  = _rectangularity_score(quad)
    return float(area * (rect_s ** 1.5) * (angle_s ** 1.5))

# ==============================
# Ön İşleme Varyantları
# ==============================

def _preprocess_normal(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def _preprocess_aggressive(bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = _unsharp_mask(g, (3,3), 1.2)
    g = _gamma_correction(g, 1.2)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    g = clahe.apply(g)
    return g

def _preprocess_soft(bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5,5), 0)
    # adaptif threshold -> kenar için vurgulu ikili görüntü
    th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 21, 5)
    return th

def _preprocess_color_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 170]), np.array([180, 70, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    return mask

# ==============================
# Çizgi & Kesişim Tabanlı Köşe Tespiti
# ==============================

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
    L = lines[:, 0, :].astype(np.float64)
    for (x1, y1, x2, y2), (x3, y3, x4, y4) in combinations(L, 2):
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:
            continue
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
        if np.isfinite(px) and np.isfinite(py):
            px = float(np.clip(px, 0, w - 1))
            py = float(np.clip(py, 0, h - 1))
            pts.append([px, py])
    return np.array(pts, dtype=np.float32)

def _compute_corners_from_lines(image: np.ndarray, edges: np.ndarray) -> np.ndarray:
    h, w = edges.shape[:2]
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                            minLineLength=max(40, min(w, h) // 5), maxLineGap=12)
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

# ==============================
# Kontur Tabanlı Tespit + Fallback
# ==============================

def _approx_quad_from_contour(c: np.ndarray) -> np.ndarray:
    """ Konturdan en iyi 4-köşe yaklaştırma. 4 çıkmazsa minAreaRect'ten döner. """
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        return approx.reshape(4, 2).astype(np.float32)
    # minAreaRect fallback
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    return box.astype(np.float32)

def _detect_by_contours(image: np.ndarray, edge_like: np.ndarray) -> Optional[np.ndarray]:
    h, w = image.shape[:2]
    edges = edge_like.copy()
    if edges.ndim == 3:
        edges = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)

    # Canny ya da ikili görüntü üzerinden morfoloji
    if edges.dtype != np.uint8:
        edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts[:12]:
        quad = _approx_quad_from_contour(c)
        if _is_valid_quad(quad, w, h):
            return quad
    return None

def _largest_box_fallback(image: np.ndarray, edges: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        box = _approx_quad_from_contour(c)
        if _is_valid_quad(box, w, h):
            return box
    return _full_image_quad(image)

# ==============================
# Çoklu Varyant Ensemble + Ölçekli İşleme
# ==============================

def _detect_document(image: np.ndarray) -> np.ndarray:
    """ Çoklu varyant, adaptif parametreler, skorlamalı seçim ve fallback. """
    H, W = image.shape[:2]

    # İşleme hız ve stabilite için ölçekle (yalnızca tespit aşamasında)
    max_side = 1200
    scale = 1.0
    proc = image
    if max(H, W) > max_side:
        scale = max_side / float(max(H, W))
        proc = cv2.resize(image, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_AREA)

    h, w = proc.shape[:2]
    gray_small = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    brightness = _estimate_brightness(gray_small)
    blur_var   = _estimate_blur(gray_small)
    canny_low, canny_high = _adaptive_canny_params(h, w, brightness, blur_var)

    # Varyant listesi
    variants = [
        ("normal",    _preprocess_normal(proc)),
        ("aggressive",_preprocess_aggressive(proc)),
        ("soft",      _preprocess_soft(proc)),
        ("color",     _preprocess_color_mask(proc)),
    ]

    candidates: List[Tuple[float, np.ndarray, str]] = []

    for name, pre in variants:
        # Kenar görüntüsü hazırla
        if pre.ndim == 3:
            pre_gray = cv2.cvtColor(pre, cv2.COLOR_BGR2GRAY)
        else:
            pre_gray = pre

        # Canny (soft varyant zaten ikili; yine de Canny denemek faydalı)
        edges = cv2.Canny(pre_gray, canny_low, canny_high)

        # 1) Hough + kesişim
        quad1 = _compute_corners_from_lines(proc, edges)
        if _is_valid_quad(quad1, w, h):
            candidates.append((_score_quad(quad1, w, h), quad1, f"hough-{name}"))

        # 2) Kontur tabanlı
        quad2 = _detect_by_contours(proc, edges) or _detect_by_contours(proc, pre_gray)
        if quad2 is not None and _is_valid_quad(quad2, w, h):
            candidates.append((_score_quad(quad2, w, h), quad2, f"contour-{name}"))

        # 3) MinAreaRect üzerinden geniş fallback aday (aday olarak ekle, en kötü ihtimal)
        if quad2 is None:
            fallback_box = _largest_box_fallback(proc, edges)
            if _is_valid_quad(fallback_box, w, h):
                candidates.append((_score_quad(fallback_box, w, h), fallback_box, f"minarea-{name}"))

    if not candidates:
        # Ölçek geri alırken full image quad
        return _full_image_quad(image)

    # En iyi adayı seç
    best_score, best_quad_small, source_tag = max(candidates, key=lambda t: t[0])

    # Orijinal ölçeye geri ölçekle
    if scale != 1.0:
        inv = 1.0 / scale
        best_quad = (np.asarray(best_quad_small, dtype=np.float32) * inv).astype(np.float32)
    else:
        best_quad = np.asarray(best_quad_small, dtype=np.float32)

    # Son bir güvenlik: sınır içi kırp ve sıralama
    H, W = image.shape[:2]
    best_quad[:, 0] = np.clip(best_quad[:, 0], 0, W-1)
    best_quad[:, 1] = np.clip(best_quad[:, 1], 0, H-1)

    return best_quad.astype(np.float32)

# ==============================
# Component
# ==============================

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
        # Görüntüyü yükle
        img_obj = Image.get_frame(img=self.image, redis_db=self.redis_db)
        if img_obj is None or img_obj.value is None:
            raise ValueError("No input image provided or failed to load.")
        src_img = self._prepare_image(img_obj.value)

        # Belge tespiti
        best_quad = _detect_document(src_img)

        # Eğer skor/validasyon şüpheliyse bir kez daha yumuşak varyantla deneriz (son şans)
        if not _is_valid_quad(best_quad, src_img.shape[1], src_img.shape[0]):
            soft = _preprocess_soft(src_img)
            edges = cv2.Canny(soft if soft.ndim == 2 else cv2.cvtColor(soft, cv2.COLOR_BGR2GRAY), 30, 90)
            alt_quad = _largest_box_fallback(src_img, edges)
            if _is_valid_quad(alt_quad, src_img.shape[1], src_img.shape[0]):
                best_quad = alt_quad
            else:
                best_quad = _full_image_quad(src_img)

        # Perspektif düzelt
        warped = _four_point_transform(src_img, best_quad)

        # Sonuçları yaz
        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)

        self.context["src_quad"] = np.asarray(best_quad, dtype=float).tolist()
        self.context["output_size"] = [int(warped.shape[1]), int(warped.shape[0])]
        return build_response(context=self)

# ==============================
# Executor
# ==============================

if __name__ == "__main__":
    Executor(sys.argv[1]).run()
