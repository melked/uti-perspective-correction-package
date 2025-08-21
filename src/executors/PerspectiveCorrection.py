import os
import sys
import cv2
import numpy as np
import math
from typing import Optional, List, Tuple, Dict

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveCorrection.src.utils.response import build_response
from components.PerspectiveCorrection.src.models.PackageModel import PackageModel


# -----------------------------------------------------------------------------
# 1) PARAMETRELER
# -----------------------------------------------------------------------------
class Params:
    def __init__(self, config=None):
        config = config or {}
        # Çözünürlük ve netlik
        self.resize_longest_edge = config.get("resize_longest_edge", 1400)
        self.unsharp_strength = float(config.get("unsharp_strength", 1.2))

        # Aydınlatma/kontrast
        self.use_illum_correction = bool(config.get("use_illum_correction", True))
        self.clahe_clip = float(config.get("clahe_clip", 2.5))
        self.clahe_grid = tuple(config.get("clahe_grid", (8, 8)))

        # Skor/filtre eşikleri
        self.score_min_area_ratio = float(config.get("score_min_area_ratio", 0.07))
        self.score_max_area_ratio = float(config.get("score_max_area_ratio", 0.985))
        self.approx_poly_epsilon_ratio = float(config.get("approx_poly_epsilon_ratio", 0.02))
        self.min_confidence_threshold = float(config.get("min_confidence_threshold", 0.28))

        # Hough parametreleri
        self.hough_line_threshold = int(config.get("hough_line_threshold", 60))
        self.hough_min_line_length = int(config.get("hough_min_line_length", 80))
        self.hough_max_line_gap = int(config.get("hough_max_line_gap", 25))
        self.hough_angle_tolerance = float(config.get("hough_angle_tolerance", 12.0))

        # Çıkış modu: "color", "gray", "binary"
        self.output_mode = str(config.get("output_mode", "color")).lower()
        self.binary_block = int(config.get("binary_block", 15))  # tek sayı
        self.binary_C = int(config.get("binary_C", 8))


# -----------------------------------------------------------------------------
# 2) YARDIMCI & GEOMETRİ
# -----------------------------------------------------------------------------
def _order_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype("float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]        # TL
    rect[2] = pts[np.argmax(s)]        # BR
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]     # TR
    rect[3] = pts[np.argmax(diff)]     # BL
    return rect


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> Optional[np.ndarray]:
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxWidth = int(max(widthA, widthB))
    maxHeight = int(max(heightA, heightB))
    if maxWidth <= 10 or maxHeight <= 10:
        return None
    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32"
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LANCZOS4)


def _unsharp_mask(image: np.ndarray, strength: float) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), 2.5)
    return cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)


def _illumination_correction(gray: np.ndarray) -> np.ndarray:
    """
    Basit shade removal: büyük kernel opening ile arka plan tahmini, ardından normalize.
    """
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, se)
    corrected = cv2.divide(gray, cv2.max(background, 1), scale=255)
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    return corrected


def _apply_clahe(gray: np.ndarray, clip: float, grid: Tuple[int, int]) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    return clahe.apply(gray)


def _line_intersection(line1, line2) -> Optional[Tuple[int, int]]:
    x1, y1, x2, y2 = line1[0]
    x3, y3, x4, y4 = line2[0]
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    t_num = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    t = t_num / denom
    ix = int(round(x1 + t * (x2 - x1)))
    iy = int(round(y1 + t * (y2 - y1)))
    return (ix, iy)


# -----------------------------------------------------------------------------
# 3) QUAD ADAYLARI
# -----------------------------------------------------------------------------
def _get_quads_from_contours(contours: List[np.ndarray], params: Params, img_shape: Tuple[int, int]) -> List[np.ndarray]:
    quads: List[np.ndarray] = []
    h, w = img_shape[:2]
    min_area = params.score_min_area_ratio * (w * h)

    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        if cv2.contourArea(c) < min_area:
            continue

        # Önce approx ile 4'lü poligon dene
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, params.approx_poly_epsilon_ratio * peri, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            quads.append(approx.reshape(4, 2).astype(np.float32))
            continue

        # Yedek: minAreaRect → boxPoints (her koşulda 4 nokta verir)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = box.astype(np.float32)
        if cv2.contourArea(box.astype(np.int32)) >= min_area:
            quads.append(box)

    return quads


def strategy_canny(image: np.ndarray, params: Params, is_dark_doc: bool = False, low_contrast: bool = False) -> List[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if params.use_illum_correction:
        gray = _illumination_correction(gray)

    if low_contrast:
        gray = _apply_clahe(gray, params.clahe_clip, params.clahe_grid)

    # Kenar korumalı yumuşatma
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)

    # Otomatik eşik (median) + koyu belge için hafif sapma
    v = np.median(blurred)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    if is_dark_doc:
        lower = max(10, lower - 10)
        upper = min(255, upper - 10)

    edges = cv2.Canny(blurred, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return _get_quads_from_contours(contours, params, image.shape)


def strategy_hough_lines(image: np.ndarray, params: Params) -> List[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if params.use_illum_correction:
        gray = _illumination_correction(gray)

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=params.hough_line_threshold,
        minLineLength=params.hough_min_line_length,
        maxLineGap=params.hough_max_line_gap
    )
    if lines is None or len(lines) < 4:
        return []

    # Yatay/dikey ayır
    horizontal, vertical = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        if angle < params.hough_angle_tolerance or abs(angle - 180) < params.hough_angle_tolerance:
            horizontal.append(line)
        elif abs(angle - 90) < params.hough_angle_tolerance:
            vertical.append(line)

    if len(horizontal) < 2 or len(vertical) < 2:
        return []

    # En üst/alt ve sol/sağ uç çizgileri seç
    horizontal.sort(key=lambda ln: (min(ln[0][1], ln[0][3])))
    vertical.sort(key=lambda ln: (min(ln[0][0], ln[0][2])))

    top_line, bottom_line = horizontal[0], horizontal[-1]
    left_line, right_line = vertical[0], vertical[-1]

    tl = _line_intersection(top_line, left_line)
    tr = _line_intersection(top_line, right_line)
    bl = _line_intersection(bottom_line, left_line)
    br = _line_intersection(bottom_line, right_line)

    quad = None
    if all([tl, tr, br, bl]):
        quad = np.array([tl, tr, br, bl], dtype=np.float32)

    return [quad] if quad is not None else []


# -----------------------------------------------------------------------------
# 4) SKORLAMA
# -----------------------------------------------------------------------------
def _orthogonality_score(rect: np.ndarray) -> float:
    rect = _order_points(rect)
    v01 = rect[1] - rect[0]
    v12 = rect[2] - rect[1]
    v23 = rect[3] - rect[2]
    v30 = rect[0] - rect[3]
    def ang(a, b):
        na = np.linalg.norm(a); nb = np.linalg.norm(b)
        if na < 1e-6 or nb < 1e-6: return 0.0
        cosv = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
        return abs(90.0 - np.degrees(np.arccos(cosv)))
    # 90 dereceye yakınlık (0 hata → 1 puan)
    errs = np.array([ang(v01, v12), ang(v12, v23), ang(v23, v30), ang(v30, v01)])
    return float(np.clip(1.0 - (errs.mean() / 45.0), 0.0, 1.0))


def _score_quad(quad: np.ndarray, params: Params, image_shape: tuple) -> float:
    h, w = image_shape[:2]
    total_area = float(w * h)
    contour = quad.astype(np.int32)
    area = float(cv2.contourArea(contour))
    aratio = area / total_area if total_area > 0 else 0.0
    if not (params.score_min_area_ratio < aratio < params.score_max_area_ratio):
        return 0.0

    # merkezilik
    M = cv2.moments(contour)
    if abs(M.get("m00", 0.0)) < 1e-6:
        return 0.0
    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    centrality = 1.0 - (np.hypot(cx - w / 2, cy - h / 2) / (max(w, h) / 2.0))
    centrality = float(np.clip(centrality, 0.0, 1.0))

    # en/boy stabilitesi
    rect = _order_points(quad)
    width = (np.linalg.norm(rect[1] - rect[0]) + np.linalg.norm(rect[2] - rect[3])) / 2.0
    height = (np.linalg.norm(rect[3] - rect[0]) + np.linalg.norm(rect[2] - rect[1])) / 2.0
    if min(width, height) < 1.0:
        return 0.0
    ratio = max(width, height) / max(1.0, min(width, height))
    ratio_score = float(np.clip(1.5 / ratio, 0.0, 1.0))  # 1.0–1.5 iyi, üstü kademeli düşer

    ortho = _orthogonality_score(rect)

    # bileşik skor
    return 0.45 * centrality + 0.30 * aratio + 0.15 * ortho + 0.10 * ratio_score


# -----------------------------------------------------------------------------
# 5) ANA BİLEŞEN
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
        if img is None or img.size == 0:
            raise ValueError("Input image is empty or None.")

        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def _diagnose_image(self, image: np.ndarray) -> Dict:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean = float(np.mean(gray))
        std_dev = float(np.std(gray))
        return {
            "brightness": "dark" if mean < 85 else "bright" if mean > 170 else "normal",
            "contrast": "low" if std_dev < 40 else "normal",
            "is_dark_doc": mean < 128
        }

    def _final_enhance(self, warped_bgr: np.ndarray) -> np.ndarray:
        mode = self.params.output_mode
        if mode == "color":
            out = _unsharp_mask(warped_bgr, self.params.unsharp_strength)
            return out
        gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
        if mode == "gray":
            gray = _apply_clahe(gray, self.params.clahe_clip, self.params.clahe_grid)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        # binary
        block = self.params.binary_block if self.params.binary_block % 2 == 1 else self.params.binary_block + 1
        bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, block, self.params.binary_C)
        return cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)

    def run(self):
        img_obj = Image.get_frame(img=self.image, redis_db=self.redis_db)
        if img_obj is None or img_obj.value is None:
            raise ValueError("No input image provided or failed to load.")

        src_img_orig = self._prepare_image(img_obj.value)
        h, w = src_img_orig.shape[:2]

        # Ölçekleme
        longest = max(h, w)
        scale = self.params.resize_longest_edge / float(longest) if longest > self.params.resize_longest_edge else 1.0
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        work_img = cv2.resize(src_img_orig, (int(round(w * scale)), int(round(h * scale))), interpolation=interp)

        # Hafif netlik
        work_img = _unsharp_mask(work_img, self.params.unsharp_strength * 0.5)

        # 1) Teşhis
        diagnostics = self._diagnose_image(work_img)

        # 2) Strateji sırası
        if diagnostics["contrast"] == "low":
            strategy_pipeline = [
                ("hough_lines", strategy_hough_lines, {}),
                ("canny", strategy_canny, {"is_dark_doc": diagnostics["is_dark_doc"], "low_contrast": True}),
            ]
        else:
            strategy_pipeline = [
                ("canny", strategy_canny, {"is_dark_doc": diagnostics["is_dark_doc"], "low_contrast": False}),
                ("hough_lines", strategy_hough_lines, {}),
            ]

        # 3) Stratejileri çalıştır
        all_candidates: List[np.ndarray] = []
        used_strategies = []
        for name, func, kwargs in strategy_pipeline:
            if name == "canny":
                quads = func(work_img, self.params, **kwargs)
            else:
                quads = func(work_img, self.params)
            if quads:
                used_strategies.append(name)
                all_candidates.extend(quads)

        # 4) Kontur fallback (her koşulda bir kez daha dene)
        if not all_candidates:
            gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)
            if self.params.use_illum_correction:
                gray = _illumination_correction(gray)
            thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, 15, 5)
            thr = cv2.medianBlur(thr, 3)
            contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            fallback_quads = _get_quads_from_contours(contours, self.params, work_img.shape)
            if fallback_quads:
                used_strategies.append("contour_fallback")
                all_candidates.extend(fallback_quads)

        document_quad = None
        best_score = 0.0

        # 5) En iyi adayı seç
        if all_candidates:
            scored = [(_score_quad(q, self.params, work_img.shape), q) for q in all_candidates]
            scored.sort(key=lambda x: x[0], reverse=True)
            best_score, best_quad = scored[0]
            if best_score >= self.params.min_confidence_threshold:
                document_quad = best_quad / scale  # orijinale ölçek geri
        # 6) Warp + çıktı iyileştirme
        if document_quad is not None:
            warped = _four_point_transform(src_img_orig, document_quad)
        else:
            warped = None

        if warped is None:
            warped = src_img_orig.copy()
            document_quad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

        enhanced = self._final_enhance(warped)

        # 7) Çıkış
        img_obj.value = enhanced
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)

        self.context["src_quad"] = document_quad.tolist()
        self.context["output_size"] = [enhanced.shape[1], enhanced.shape[0]]
        self.context["best_score"] = float(best_score)
        self.context["used_strategies"] = used_strategies

        return build_response(context=self)


# -----------------------------------------------------------------------------
# 6) ÇALIŞTIRICI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    Executor(sys.argv[1]).run()
