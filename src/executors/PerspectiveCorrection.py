import os
import sys
import cv2
import numpy as np
from typing import List, Tuple, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveCorrection.src.utils.response import build_response
from components.PerspectiveCorrection.src.models.PackageModel import PackageModel

def _order_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]   # tl
    rect[2] = pts[np.argmax(s)]   # br
    rect[1] = pts[np.argmin(diff)]  # tr
    rect[3] = pts[np.argmax(diff)]  # bl
    return rect


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _order_points(pts)
    tl, tr, br, bl = rect
    maxWidth = int(round(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))))
    maxHeight = int(round(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))))
    dst = np.array([[0, 0], [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LANCZOS4)


def _full_image_quad(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)


# ==============================
# Ön işleme (agresif zincir)
# ==============================
def _contrast_enhance(gray: np.ndarray) -> np.ndarray:
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)
    # Aydınlık dengesizliği bastırma (morf. opening ile aydınlık arka planı çıkar)
    bg = cv2.morphologyEx(g, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    g = cv2.subtract(g, bg)
    return g


def _edges_strong(gray: np.ndarray) -> np.ndarray:
    # Kenar korumalı yumuşatma
    smooth = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # Top-hat & Black-hat ile belge–zemin farkını öne çıkar
    k15 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    top = cv2.morphologyEx(smooth, cv2.MORPH_TOPHAT, k15)
    blk = cv2.morphologyEx(smooth, cv2.MORPH_BLACKHAT, k15)
    enh = cv2.add(smooth, top)
    enh = cv2.subtract(enh, blk)

    # Scharr gradyan (daha güçlü)
    gx = cv2.Scharr(enh, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(enh, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Adaptif ikilendirme + Canny birleştir
    thr = cv2.adaptiveThreshold(enh, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 21, -5)
    can = cv2.Canny(mag, 30, 180)

    # Birlikte (OR) → sonra güçlü kapama
    edges = cv2.bitwise_or(thr, can)
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k_close)
    # İnce tüyleri atmak için küçük açma
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    return edges


def _preprocess_variants(bgr: np.ndarray) -> List[np.ndarray]:
    """Aşırı karmaşık zeminler için birkaç varyant döndürür; tekrar yok, parametre setleri var."""
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # 1) Ana zincir
    g1 = _contrast_enhance(gray)
    e1 = _edges_strong(g1)

    # 2) Gamma koyulaştır -> kenarı öne çıkar
    mean = float(np.mean(gray))
    gamma = 1.6 if mean < 100 else 0.8 if mean > 170 else 1.0
    if gamma != 1.0:
        inv = 1.0 / gamma
        table = (np.arange(256) / 255.0) ** inv * 255.0
        g2 = cv2.LUT(gray, table.astype(np.uint8))
    else:
        g2 = gray
    g2 = _contrast_enhance(g2)
    e2 = _edges_strong(g2)

    # 3) Yüksek frekans vurgusu (unsharp)
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    sharp = cv2.addWeighted(gray, 1.8, blur, -0.8, 0)
    g3 = _contrast_enhance(sharp)
    e3 = _edges_strong(g3)

    return [e1, e2, e3]


# ==============================
# Dörtgen bulma – Kontur ve Hough birlikte
# ==============================
def _find_quad_from_contours(mask: np.ndarray, ref: np.ndarray,
                             min_area_ratio: float = 0.05) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    img_area = ref.shape[0] * ref.shape[1]
    min_area = img_area * min_area_ratio

    for c in contours[:10]:  # ilk 10 büyük kontur yeter
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            # dikdörtgene yakınlık (rektangularite)
            rect = cv2.minAreaRect(approx)
            box = cv2.boxPoints(rect).astype(np.float32)
            rect_area = cv2.contourArea(box)
            if rect_area <= 0:
                continue
            rectangularity = float(area) / rect_area
            if rectangularity > 0.7:
                return approx.reshape(4, 2).astype(np.float32)

    # Olmadıysa minAreaRect dönüşümü
    rect = cv2.minAreaRect(contours[0])
    box = cv2.boxPoints(rect).astype(np.float32)
    if cv2.contourArea(box) >= min_area:
        return box
    return None


def _lines_from_hough(mask: np.ndarray) -> Optional[np.ndarray]:
    lines = cv2.HoughLinesP(mask, 1, np.pi / 180,
                            threshold=140, minLineLength=0.25 * min(mask.shape[:2]),
                            maxLineGap=12)
    if lines is None:
        return None
    return lines[:, 0, :]  # (N,4)


def _merge_and_select_lines(lines: np.ndarray,
                            angle_eps_deg: float = 12.0) -> Optional[List[Tuple[float, float]]]:
    """
    Hough segments -> iki yaklaşık dikey ve iki yaklaşık yatay sonsuz doğru seç.
    Her doğru (a,b,c) yerine (rho, theta) polar döner:  x*cosθ + y*sinθ = ρ
    """
    if lines is None or len(lines) < 2:
        return None

    def seg_to_polar(x1, y1, x2, y2) -> Tuple[float, float]:
        dx, dy = x2 - x1, y2 - y1
        theta = np.arctan2(dy, dx)  # segment yönü
        # doğru normali için +90°
        theta_n = (theta + np.pi / 2.0)
        # ρ = x*cosθ + y*sinθ (ilk noktadan)
        rho = x1 * np.cos(theta_n) + y1 * np.sin(theta_n)
        # ρ pozitif olsun, olmazsa (ρ, θ+π)
        if rho < 0:
            rho = -rho
            theta_n += np.pi
        # normalize
        theta_n = (theta_n + 2 * np.pi) % (2 * np.pi)
        return float(rho), float(theta_n)

    polars = np.array([seg_to_polar(*l) for l in lines], dtype=np.float32)

    # yatay grubu: theta ~ 0 veya π ; dikey grubu: theta ~ π/2
    def angle_group(theta):
        t = (theta + np.pi) % np.pi  # [0, π)
        return 'h' if (t < np.deg2rad(angle_eps_deg) or abs(t - np.pi) < np.deg2rad(angle_eps_deg)) else \
               ('v' if abs(t - np.pi/2) < np.deg2rad(angle_eps_deg) else 'o')

    groups = {'h': [], 'v': []}
    for (rho, theta) in polars:
        g = angle_group(theta)
        if g in groups:
            groups[g].append((rho, theta))

    if len(groups['h']) < 1 or len(groups['v']) < 1:
        return None

    # her grupta uç/lateral iki doğruyu seç: ρ min ve ρ max
    def extremes(arr):
        arr = sorted(arr, key=lambda x: x[0])
        return [arr[0], arr[-1]] if len(arr) >= 2 else arr

    hs = extremes(groups['h'])
    vs = extremes(groups['v'])
    if len(hs) < 2 or len(vs) < 2:
        return None

    return hs + vs  # [h_min, h_max, v_min, v_max] – her biri (rho, theta)


def _intersect_rhotheta(l1: Tuple[float, float], l2: Tuple[float, float]) -> Optional[Tuple[float, float]]:
    # x cosθ + y sinθ = ρ
    rho1, th1 = l1
    rho2, th2 = l2
    a1, b1 = np.cos(th1), np.sin(th1)
    a2, b2 = np.cos(th2), np.sin(th2)
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-6:
        return None
    x = (rho1 * b2 - rho2 * b1) / det
    y = (a1 * rho2 - a2 * rho1) / det
    return (float(x), float(y))


def _find_quad_from_hough(mask: np.ndarray, ref: np.ndarray) -> Optional[np.ndarray]:
    lines = _lines_from_hough(mask)
    merged = _merge_and_select_lines(lines)
    if not merged:
        return None
    h1, h2, v1, v2 = merged
    pts = [
        _intersect_rhotheta(h1, v1),
        _intersect_rhotheta(h1, v2),
        _intersect_rhotheta(h2, v2),
        _intersect_rhotheta(h2, v1),
    ]
    if any(p is None for p in pts):
        return None
    pts = np.array(pts, dtype=np.float32)

    # görüntü sınırları içinde mi?
    h, w = ref.shape[:2]
    if not np.all((pts[:, 0] >= -5) & (pts[:, 0] <= w + 5) &
                  (pts[:, 1] >= -5) & (pts[:, 1] <= h + 5)):
        # dışarı taşsa da deneyelim; sonra alan filtresi
        pass

    # konveks + yeterli alan
    hull = cv2.convexHull(pts)
    if len(hull) < 4:
        return None
    quad = hull[:4].reshape(4, 2)
    if cv2.contourArea(quad) < 0.05 * (h * w):
        return None
    return quad.astype(np.float32)


def _score_quad(image: np.ndarray, quad: np.ndarray) -> float:
    h, w = image.shape[:2]
    area = cv2.contourArea(quad.astype(np.float32))
    if area <= 0:
        return -1.0
    # merkeze yakınlık
    cx, cy = np.mean(quad, axis=0)
    center_penalty = (abs(cx - w / 2) / w + abs(cy - h / 2) / h) / 2.0
    center_score = 1.0 - center_penalty
    # dikdörtgene yakınlık (minAreaRect alan oranı)
    rect = cv2.minAreaRect(quad.astype(np.float32))
    box = cv2.boxPoints(rect).astype(np.float32)
    rect_area = max(cv2.contourArea(box), 1.0)
    rectangularity = float(area) / rect_area  # 0..1
    return area * (0.6 * center_score + 0.4 * rectangularity)


def _detect_document_quad(bgr: np.ndarray) -> np.ndarray:
    masks = _preprocess_variants(bgr)
    candidates: List[np.ndarray] = []

    for m in masks:
        q1 = _find_quad_from_contours(m, bgr, min_area_ratio=0.04)
        if q1 is not None:
            candidates.append(q1)
        q2 = _find_quad_from_hough(m, bgr)
        if q2 is not None:
            candidates.append(q2)

    # aday yoksa tüm görüntü
    if not candidates:
        return _full_image_quad(bgr)

    # benzerleri ele (yakın köşeleri aynı say)
    uniq: List[np.ndarray] = []
    for q in candidates:
        if not any(np.allclose(_order_points(q), _order_points(u), atol=4.0) for u in uniq):
            uniq.append(q)

    # en iyi skor
    best = max(uniq, key=lambda q: _score_quad(bgr, q))
    return best.astype(np.float32)

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

        # 1) Dörtgen tespit
        best_quad = _detect_document_quad(src_img)

        # 2) Perspektif düzeltme
        warped = _four_point_transform(src_img, best_quad)

        # çıktı yaz
        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)

        self.context["src_quad"] = _order_points(best_quad).tolist()
        self.context["output_size"] = [warped.shape[1], warped.shape[0]]

        return build_response(context=self)

if __name__ == "__main__":
    Executor(sys.argv[1]).run()
