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
# 1. PARAMETRE YÖNETİMİ SINIFI (GELİŞTİRİLMİŞ)
# -----------------------------------------------------------------------------
class Params:
    def __init__(self, config=None):
        config = config or {}
        self.resize_longest_edge = config.get("resize_longest_edge", 1000)
        self.unsharp_strength = config.get("unsharp_strength", 1.5)
        self.score_min_area_ratio = config.get("score_min_area_ratio", 0.10)
        self.score_max_area_ratio = config.get("score_max_area_ratio", 0.98)
        self.approx_poly_epsilon_ratio = config.get("approx_poly_epsilon_ratio", 0.02)
        self.min_confidence_threshold = config.get("min_confidence_threshold", 0.30)
        self.hough_line_threshold = config.get("hough_line_threshold", 50)
        self.hough_min_line_length = config.get("hough_min_line_length", 50)
        self.hough_max_line_gap = config.get("hough_max_line_gap", 20)
        self.hough_angle_tolerance = config.get("hough_angle_tolerance", 10)

        # YENİ PARAMETRELER - Robustluk için
        self.multi_scale_detection = config.get("multi_scale_detection", True)
        self.adaptive_threshold = config.get("adaptive_threshold", True)
        self.shadow_compensation = config.get("shadow_compensation", True)
        self.glare_detection = config.get("glare_detection", True)
        self.rotation_invariant = config.get("rotation_invariant", True)
        self.edge_refinement = config.get("edge_refinement", True)
        self.noise_reduction_strength = config.get("noise_reduction_strength", 2)


# -----------------------------------------------------------------------------
# 2. YENİ YARDIMCI FONKSİYONLAR - ROBUST DETECTION
# -----------------------------------------------------------------------------
def _detect_glare_regions(image: np.ndarray) -> np.ndarray:
    """Parlama bölgelerini tespit eder ve maskeler"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Çok parlak bölgeleri tespit et
    _, glare_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    # Maskeyi genişlet
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    glare_mask = cv2.morphologyEx(glare_mask, cv2.MORPH_DILATE, kernel)
    return glare_mask


def _compensate_shadows(image: np.ndarray) -> np.ndarray:
    """Gölge kompansasyonu yapar"""
    # LAB renk uzayında gölge düzeltme
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # L kanalında CLAHE uygula
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    # Tekrar birleştir
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _adaptive_preprocessing(image: np.ndarray, diagnostics: Dict) -> np.ndarray:
    """Görüntü koşullarına göre adaptif ön işleme"""
    processed = image.copy()

    # Gürültü azaltma - koşullara göre adaptif
    if diagnostics.get("noise_level", "normal") == "high":
        processed = cv2.bilateralFilter(processed, 15, 80, 80)
    else:
        processed = cv2.bilateralFilter(processed, 9, 75, 75)

    # Gölge kompansasyonu
    if diagnostics.get("has_shadows", False):
        processed = _compensate_shadows(processed)

    # Kontrast iyileştirme
    if diagnostics.get("contrast", "normal") == "low":
        processed = cv2.convertScaleAbs(processed, alpha=1.3, beta=10)

    return processed


def _multi_scale_contour_detection(image: np.ndarray, params: Params) -> List[np.ndarray]:
    """Çoklu ölçekte kontur tespiti"""
    all_contours = []
    scales = [1.0, 0.8, 1.2] if params.multi_scale_detection else [1.0]

    for scale in scales:
        if scale != 1.0:
            h, w = image.shape[:2]
            scaled_img = cv2.resize(image, (int(w * scale), int(h * scale)))
        else:
            scaled_img = image.copy()

        gray = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2GRAY)

        # Adaptif eşikleme
        if params.adaptive_threshold:
            # Hem global hem adaptif eşikleme kombinasyonu
            _, binary1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
            combined = cv2.bitwise_and(binary1, binary2)
        else:
            combined = cv2.Canny(gray, 50, 150)

        # Morfolojik operasyonlar
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Ölçeği geri çevir
        if scale != 1.0:
            for contour in contours:
                contour[:, :, 0] = (contour[:, :, 0] / scale).astype(np.int32)
                contour[:, :, 1] = (contour[:, :, 1] / scale).astype(np.int32)

        all_contours.extend(contours)

    return all_contours


def _refine_quad_edges(quad: np.ndarray, image: np.ndarray) -> np.ndarray:
    """Dörtgen kenarlarını ince ayar yapar"""
    refined_quad = quad.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for i in range(4):
        # Her köşe için yerel gradyan maksimumunu ara
        corner = quad[i].astype(int)
        search_radius = 10

        y_min = max(0, corner[1] - search_radius)
        y_max = min(gray.shape[0], corner[1] + search_radius)
        x_min = max(0, corner[0] - search_radius)
        x_max = min(gray.shape[1], corner[0] + search_radius)

        roi = gray[y_min:y_max, x_min:x_max]
        if roi.size > 0:
            # Gradyan büyüklüğünü hesapla
            grad_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

            # Maksimum gradyan noktasını bul
            max_loc = np.unravel_index(np.argmax(magnitude), magnitude.shape)
            refined_corner = np.array([x_min + max_loc[1], y_min + max_loc[0]])

            # Orijinal köşeden çok uzak değilse güncelle
            if np.linalg.norm(refined_corner - corner) < search_radius:
                refined_quad[i] = refined_corner

    return refined_quad


def _rotation_invariant_detection(image: np.ndarray, params: Params) -> List[np.ndarray]:
    """Rotasyon değişmez tespit"""
    if not params.rotation_invariant:
        return []

    quads = []
    # Küçük rotasyonlar dene
    angles = [-5, 5, -10, 10] if params.rotation_invariant else [0]

    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    for angle in angles:
        # Görüntüyü döndür
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        # Normal tespit yap
        candidates = strategy_canny(rotated, params)

        # Bulunan dörtgenleri geri döndür
        if candidates:
            inverse_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
            for quad in candidates:
                # Homojen koordinatlara çevir
                ones = np.ones((quad.shape[0], 1))
                quad_homo = np.hstack([quad, ones])

                # Geri döndürme dönüşümünü uygula
                rotated_back = np.dot(inverse_matrix, quad_homo.T).T
                quads.append(rotated_back.astype(np.float32))

    return quads


# -----------------------------------------------------------------------------
# 3. MEVCUT YARDIMCI & GEOMETRİ FONKSİYONLARI
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
    x1, y1, x2, y2 = line1[0]
    x3, y3, x4, y4 = line2[0]
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0: return None
    t_num = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    u_num = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3))
    t = t_num / denom
    u = u_num / denom
    ix = int(x1 + t * (x2 - x1))
    iy = int(y1 + t * (y2 - y1))
    return (ix, iy)


# -----------------------------------------------------------------------------
# 4. GELİŞTİRİLMİŞ STRATEJİLER
# -----------------------------------------------------------------------------
def _get_quads_from_contours(contours: List[np.ndarray], params: Params) -> List[np.ndarray]:
    quads = []
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:  # Daha fazla kontura bak
        peri = cv2.arcLength(c, True)
        # Adaptif epsilon - kontur boyutuna göre
        epsilon_ratio = params.approx_poly_epsilon_ratio
        if cv2.contourArea(c) < 10000:  # Küçük kontürler için daha hassas
            epsilon_ratio *= 0.5

        approx = cv2.approxPolyDP(c, epsilon_ratio * peri, True)

        # 4 köşeli ve konveks kontürları al
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad = approx.reshape(4, 2).astype(np.float32)

            # Minimum alan kontrolü
            if cv2.contourArea(quad) > 1000:
                quads.append(quad)

        # 4'ten fazla köşeli kontürları da değerlendır
        elif len(approx) > 4:
            # En büyük 4 köşeyi seç (konveks gövde kullanarak)
            hull = cv2.convexHull(approx, returnPoints=True)
            if len(hull) >= 4:
                # Douglas-Peucker ile 4 köşeye indirge
                hull_approx = cv2.approxPolyDP(hull, epsilon_ratio * cv2.arcLength(hull, True), True)
                if len(hull_approx) == 4:
                    quad = hull_approx.reshape(4, 2).astype(np.float32)
                    if cv2.contourArea(quad) > 1000:
                        quads.append(quad)

    return quads


def strategy_canny(image: np.ndarray, params: Params, is_dark_doc: bool = False) -> List[np.ndarray]:
    # Çoklu ölçekte kontur tespiti kullan
    if params.multi_scale_detection:
        contours = _multi_scale_contour_detection(image, params)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        canny_low = 30 if is_dark_doc else 50
        canny_high = 100 if is_dark_doc else 150
        edged = cv2.Canny(blurred, canny_low, canny_high)
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
                                  iterations=3)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    quads = _get_quads_from_contours(contours, params)

    # Kenar ince ayarı yap
    if params.edge_refinement and quads:
        refined_quads = []
        for quad in quads:
            refined_quad = _refine_quad_edges(quad, image)
            refined_quads.append(refined_quad)
        return refined_quads

    return quads


def strategy_hough_lines(image: np.ndarray, params: Params) -> List[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Çoklu eşik değeriyle Hough line detection
    all_lines = []
    thresholds = [params.hough_line_threshold, params.hough_line_threshold // 2, params.hough_line_threshold * 2]

    for threshold in thresholds:
        edged = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edged, 1, np.pi / 180,
                                threshold=threshold,
                                minLineLength=params.hough_min_line_length,
                                maxLineGap=params.hough_max_line_gap)
        if lines is not None:
            all_lines.extend(lines)

    if not all_lines:
        return []

    horizontal, vertical = [], []
    for line in all_lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))

        if angle < params.hough_angle_tolerance or abs(angle - 180) < params.hough_angle_tolerance:
            horizontal.append(line)
        elif abs(angle - 90) < params.hough_angle_tolerance:
            vertical.append(line)

    if len(horizontal) < 2 or len(vertical) < 2:
        return []

    # Çizgileri grupla ve en uygun olanları seç
    horizontal.sort(key=lambda line: line[0][1])
    vertical.sort(key=lambda line: line[0][0])

    # En dış çizgileri al
    top_line, bottom_line = horizontal[0], horizontal[-1]
    left_line, right_line = vertical[0], vertical[-1]

    # Kesişim noktalarını hesapla
    tl = _line_intersection(top_line, left_line)
    tr = _line_intersection(top_line, right_line)
    bl = _line_intersection(bottom_line, left_line)
    br = _line_intersection(bottom_line, right_line)

    if all((tl, tr, bl, br)):
        quad = np.array([tl, tr, br, bl], dtype=np.float32)
        return [quad]

    return []


# YENİ STRATEJİ: Corner Detection tabanlı
def strategy_corner_detection(image: np.ndarray, params: Params) -> List[np.ndarray]:
    """Harris corner detection ve RANSAC ile dörtgen tespiti"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Harris corner detection
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01,
                                      minDistance=30, blockSize=3, useHarrisDetector=True, k=0.04)

    if corners is None or len(corners) < 4:
        return []

    corners = corners.reshape(-1, 2)

    # En dış köşeleri bul (convex hull kullanarak)
    hull_indices = cv2.convexHull(corners, returnPoints=False).flatten()
    hull_points = corners[hull_indices]

    if len(hull_points) >= 4:
        # En uzak 4 köşeyi seç
        # Görüntü merkezinden en uzak köşeleri bul
        h, w = image.shape[:2]
        center = np.array([w / 2, h / 2])

        distances = np.linalg.norm(hull_points - center, axis=1)
        farthest_indices = np.argsort(distances)[-4:]
        quad_candidates = hull_points[farthest_indices]

        # Dörtgen oluşturmaya çalış
        quad = _order_points(quad_candidates)
        return [quad.astype(np.float32)]

    return []


# -----------------------------------------------------------------------------
# 5. GELİŞTİRİLMİŞ SKORLAMA SİSTEMİ
# -----------------------------------------------------------------------------
def _score_quad(quad: np.ndarray, params: Params, image_shape: tuple, diagnostics: Dict = None) -> float:
    h, w = image_shape[:2]
    total_area = w * h
    contour = quad.astype(np.int32)
    area = cv2.contourArea(contour)

    # Temel alan kontrolü
    area_ratio = area / total_area
    if not (params.score_min_area_ratio < area_ratio < params.score_max_area_ratio):
        return 0.0

    # Merkez pozisyon skoru
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return 0.0

    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    centrality = 1.0 - (np.linalg.norm(np.array([cx, cy]) - np.array([w / 2, h / 2])) / (max(w, h) / 2))

    # Geometrik özellikler
    (tl, tr, br, bl) = _order_points(quad)

    # Kenar uzunlukları
    top_length = np.linalg.norm(tr - tl)
    bottom_length = np.linalg.norm(br - bl)
    left_length = np.linalg.norm(tl - bl)
    right_length = np.linalg.norm(tr - br)

    # Paralel kenar skoru (karşılıklı kenarlar benzer uzunlukta olmalı)
    horizontal_similarity = 1.0 - abs(top_length - bottom_length) / max(top_length, bottom_length)
    vertical_similarity = 1.0 - abs(left_length - right_length) / max(left_length, right_length)
    parallel_score = (horizontal_similarity + vertical_similarity) / 2

    # Aspect ratio skoru
    width = (top_length + bottom_length) / 2
    height = (left_length + right_length) / 2
    if min(width, height) < 1:
        return 0.0

    aspect_ratio = max(width, height) / min(width, height)
    # Belge için tipik aspect ratios: A4 = 1.41, Letter = 1.29
    if 1.1 < aspect_ratio < 2.2:
        aspect_score = 1.0
    elif aspect_ratio < 3.0:
        aspect_score = 0.7
    else:
        aspect_score = 0.3

    # Açı skoru (köşelerin 90 dereceye yakınlığı)
    angles = []
    for i in range(4):
        p1 = quad[i]
        p2 = quad[(i + 1) % 4]
        p3 = quad[(i + 2) % 4]

        v1 = p1 - p2
        v2 = p3 - p2

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        angle = abs(np.degrees(np.arccos(np.clip(cos_angle, -1, 1))))
        angles.append(angle)

    angle_deviations = [abs(angle - 90) for angle in angles]
    angle_score = 1.0 - (np.mean(angle_deviations) / 90.0)
    angle_score = max(0, angle_score)

    # Konvekslik skoru
    convexity_score = 1.0 if cv2.isContourConvex(contour) else 0.3

    # Kenar kalitesi skoru (kenarların düzgünlüğü)
    perimeter = cv2.arcLength(contour, True)
    hull = cv2.convexHull(contour)
    hull_perimeter = cv2.arcLength(hull, True)
    solidity = perimeter / (hull_perimeter + 1e-10)
    edge_quality_score = min(1.0, solidity)

    # Final skor hesaplama (ağırlıklı ortalama)
    weights = {
        'centrality': 0.15,
        'area': 0.20,
        'aspect': 0.15,
        'parallel': 0.15,
        'angle': 0.20,
        'convexity': 0.10,
        'edge_quality': 0.05
    }

    final_score = (
            weights['centrality'] * centrality +
            weights['area'] * (area_ratio / params.score_max_area_ratio) +
            weights['aspect'] * aspect_score +
            weights['parallel'] * parallel_score +
            weights['angle'] * angle_score +
            weights['convexity'] * convexity_score +
            weights['edge_quality'] * edge_quality_score
    )

    return final_score


# -----------------------------------------------------------------------------
# 6. GELİŞTİRİLMİŞ TANILAMA SİSTEMİ
# -----------------------------------------------------------------------------
def _diagnose_image_advanced(image: np.ndarray) -> Dict:
    """Gelişmiş görüntü analizi"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Temel istatistikler
    mean = np.mean(gray)
    std_dev = np.std(gray)

    # Histogram analizi
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_norm = hist / (h * w)

    # Gürültü seviyesi tahmini
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Gradyan analizi
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    edge_density = np.sum(gradient_magnitude > 50) / (h * w)

    # Parlama tespiti
    bright_pixels = np.sum(gray > 240) / (h * w)

    # Gölge tespiti (histogram bimodal mı?)
    hist_smooth = cv2.GaussianBlur(hist_norm.reshape(-1, 1), (0, 0), 5).flatten()
    peaks = []
    for i in range(1, len(hist_smooth) - 1):
        if hist_smooth[i] > hist_smooth[i - 1] and hist_smooth[i] > hist_smooth[i + 1] and hist_smooth[i] > 0.001:
            peaks.append(i)

    diagnostics = {
        "brightness": "dark" if mean < 85 else "bright" if mean > 170 else "normal",
        "contrast": "low" if std_dev < 40 else "high" if std_dev > 80 else "normal",
        "noise_level": "high" if laplacian_var < 100 else "normal",
        "edge_density": "high" if edge_density > 0.1 else "low" if edge_density < 0.05 else "normal",
        "has_glare": bright_pixels > 0.05,
        "has_shadows": len(peaks) >= 2,
        "is_dark_doc": mean < 128,
        "is_complex": edge_density > 0.15 or len(peaks) > 2
    }

    print(f"Gelişmiş Görüntü Teşhisi: {diagnostics}")
    return diagnostics


# -----------------------------------------------------------------------------
# 7. ANA BİLEŞEN (GELİŞTİRİLMİŞ)
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
        """Görüntünün temel özelliklerini analiz ederek bir teşhis raporu oluşturur."""
        return _diagnose_image_advanced(image)

    def _adaptive_strategy_selection(self, diagnostics: Dict) -> List:
        """Teşhise göre en uygun strateji dizisini seçer"""
        strategies = []

        # Temel strateji seçimi
        if diagnostics.get("contrast", "normal") == "low" or diagnostics.get("has_shadows", False):
            print("Düşük kontrast/gölge tespit edildi. Hough Lines öncelikli.")
            strategies.append(strategy_hough_lines)
            strategies.append(strategy_canny)
        else:
            print("Normal kontrast tespit edildi. Canny öncelikli.")
            strategies.append(strategy_canny)
            strategies.append(strategy_hough_lines)

        # Karmaşık görüntüler için corner detection ekle
        if diagnostics.get("is_complex", False):
            print("Karmaşık görüntü tespit edildi. Corner detection ekleniyor.")
            strategies.append(strategy_corner_detection)

        # Rotasyon değişmez tespit gerekiyorsa
        if self.params.rotation_invariant:
            rotation_strategy = lambda img, params: _rotation_invariant_detection(img, params)
            strategies.append(rotation_strategy)

        return strategies

    def _ensemble_detection(self, image: np.ndarray, diagnostics: Dict) -> List[np.ndarray]:
        """Çoklu strateji kullanarak ensemble detection yapar"""

        # Önişleme - adaptif
        processed_image = _adaptive_preprocessing(image, diagnostics)

        # Strateji seçimi
        strategies = self._adaptive_strategy_selection(diagnostics)

        all_candidates = []
        strategy_scores = {}  # Her stratejiden gelen en iyi skorları takip et

        for i, strategy_func in enumerate(strategies):
            print(f"Çalıştırılan strateji {i + 1}/{len(strategies)}: {strategy_func.__name__}")

            try:
                # Stratejiye özel argümanları geçir
                if hasattr(strategy_func, '__name__') and strategy_func.__name__ == 'strategy_canny':
                    candidates = strategy_func(processed_image, self.params,
                                               is_dark_doc=diagnostics.get("is_dark_doc", False))
                else:
                    candidates = strategy_func(processed_image, self.params)

                if candidates:
                    print(f" -> {len(candidates)} aday bulundu.")

                    # Her strateji için skorları hesapla
                    strategy_best_score = 0
                    for candidate in candidates:
                        score = _score_quad(candidate, self.params, image.shape, diagnostics)
                        if score > strategy_best_score:
                            strategy_best_score = score
                        all_candidates.append((score, candidate, strategy_func.__name__))

                    strategy_scores[strategy_func.__name__] = strategy_best_score
                    print(f" -> En iyi aday skoru: {strategy_best_score:.3f}")
                else:
                    print(" -> Aday bulunamadı.")
                    strategy_scores[strategy_func.__name__] = 0.0

            except Exception as e:
                print(f" -> Strateji hatası: {str(e)}")
                strategy_scores[strategy_func.__name__] = 0.0
                continue

        # Strateji performanslarını raporla
        print(f"Strateji performansları: {strategy_scores}")

        return all_candidates

    def _consensus_filtering(self, scored_candidates: List[Tuple], threshold: float = 0.1) -> List[Tuple]:
        """Benzer dörtgenleri gruplar ve en iyilerini seçer"""
        if not scored_candidates:
            return []

        # Skorlara göre sırala
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        filtered_candidates = []
        used_indices = set()

        for i, (score1, quad1, strategy1) in enumerate(scored_candidates):
            if i in used_indices:
                continue

            # Bu dörtgen için benzer olanları bul
            similar_group = [(score1, quad1, strategy1)]
            used_indices.add(i)

            for j, (score2, quad2, strategy2) in enumerate(scored_candidates[i + 1:], i + 1):
                if j in used_indices:
                    continue

                # İki dörtgen benzer mi kontrol et
                quad1_ordered = _order_points(quad1)
                quad2_ordered = _order_points(quad2)

                # Köşeler arası ortalama mesafeyi hesapla
                distances = [np.linalg.norm(p1 - p2) for p1, p2 in zip(quad1_ordered, quad2_ordered)]
                avg_distance = np.mean(distances)

                # Görüntü boyutuna göre normalize et
                h, w = 1000, 1000  # Varsayılan boyut
                normalized_distance = avg_distance / max(h, w)

                if normalized_distance < threshold:  # Benzer dörtgenler
                    similar_group.append((score2, quad2, strategy2))
                    used_indices.add(j)

            # Grup içindeki en iyi dörtgeni seç
            if similar_group:
                best_in_group = max(similar_group, key=lambda x: x[0])
                filtered_candidates.append(best_in_group)

        return filtered_candidates

    def _fallback_detection(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Tüm stratejiler başarısız olursa fallback çözümü"""
        h, w = image.shape[:2]

        # Basit corner detection ile son deneme
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Agresif corner detection
        corners = cv2.goodFeaturesToTrack(
            gray, maxCorners=50, qualityLevel=0.005,
            minDistance=20, blockSize=3, useHarrisDetector=True, k=0.04
        )

        if corners is not None and len(corners) >= 4:
            corners = corners.reshape(-1, 2)

            # En dış 4 köşeyi bul
            # Sol üst, sağ üst, sağ alt, sol alt
            top_left = corners[np.argmin(corners[:, 0] + corners[:, 1])]
            top_right = corners[np.argmax(corners[:, 0] - corners[:, 1])]
            bottom_right = corners[np.argmax(corners[:, 0] + corners[:, 1])]
            bottom_left = corners[np.argmin(corners[:, 0] - corners[:, 1])]

            fallback_quad = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

            # Minimum alan kontrolü
            if cv2.contourArea(fallback_quad.astype(np.int32)) > (w * h * 0.1):
                print("Fallback detection ile dörtgen bulundu.")
                return fallback_quad

        # Son çare: görüntü sınırlarından %10 içeride bir dörtgen
        margin = min(w, h) * 0.1
        fallback_quad = np.array([
            [margin, margin],
            [w - margin, margin],
            [w - margin, h - margin],
            [margin, h - margin]
        ], dtype=np.float32)

        print("Tüm detection yöntemleri başarısız. Görüntü sınırları kullanılıyor.")
        return fallback_quad

    def run(self):
        # Temel hazırlık
        img_obj = Image.get_frame(img=self.image, redis_db=self.redis_db)
        if img_obj is None or img_obj.value is None:
            raise ValueError("No input image provided or failed to load.")

        src_img_orig = self._prepare_image(img_obj.value)
        h, w = src_img_orig.shape[:2]

        print(f"Orijinal görüntü boyutu: {w}x{h}")

        # Çalışma görüntüsü hazırlama
        scale = self.params.resize_longest_edge / max(h, w) if max(h, w) > self.params.resize_longest_edge else 1
        work_img = cv2.resize(src_img_orig, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        work_img = _unsharp_mask(work_img, self.params.unsharp_strength)

        print(f"Çalışma görüntüsü boyutu: {work_img.shape[1]}x{work_img.shape[0]}, Ölçek: {scale:.3f}")

        # 1. Gelişmiş görüntü teşhisi
        diagnostics = self._diagnose_image(work_img)

        # Parlama kompensasyonu
        if diagnostics.get("has_glare", False) and self.params.glare_detection:
            print("Parlama tespit edildi, kompensasyon uygulanıyor.")
            glare_mask = _detect_glare_regions(work_img)
            work_img = cv2.inpaint(work_img, glare_mask, 3, cv2.INPAINT_TELEA)

        # 2. Ensemble detection
        all_candidates = self._ensemble_detection(work_img, diagnostics)

        # 3. Consensus filtering
        if all_candidates:
            print(f"Toplam {len(all_candidates)} aday dörtgen bulundu.")
            filtered_candidates = self._consensus_filtering(all_candidates, threshold=0.08)
            print(f"Filtrelemeden sonra {len(filtered_candidates)} aday kaldı.")
        else:
            filtered_candidates = []

        # 4. En iyi adayı seç
        document_quad = None
        if filtered_candidates:
            # En yüksek skorlu adayı seç
            best_score, best_quad, best_strategy = max(filtered_candidates, key=lambda x: x[0])
            print(f"En iyi aday: {best_strategy} stratejisinden {best_score:.3f} puanla.")

            if best_score > self.params.min_confidence_threshold:
                document_quad = best_quad
                print(f"Dörtgen kabul edildi (skor: {best_score:.3f} > eşik: {self.params.min_confidence_threshold}).")
            else:
                print(
                    f"En iyi skorlu aday eşiğin altında kaldı ({best_score:.3f} < {self.params.min_confidence_threshold}).")

        # 5. Fallback detection
        if document_quad is None:
            print("Ana detection başarısız, fallback detection deneniyor...")
            document_quad = self._fallback_detection(work_img)

        # 6. Final transformation
        warped = None
        if document_quad is not None:
            # Ölçeği geri çevir
            document_quad = document_quad / scale

            print("Perspektif dönüşümü uygulanıyor...")
            warped = _four_point_transform(src_img_orig, document_quad)

            if warped is None:
                print("Perspektif dönüşümü başarısız.")
            else:
                print(f"Dönüşüm başarılı. Çıktı boyutu: {warped.shape[1]}x{warped.shape[0]}")

        # 7. Son çare
        if warped is None:
            print("Tüm işlemler başarısız. Orijinal görüntü korunuyor.")
            warped = src_img_orig
            document_quad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

        # Sonuçları kaydet
        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)
        self.context["src_quad"] = document_quad.tolist()
        self.context["output_size"] = [warped.shape[1], warped.shape[0]]
        self.context["diagnostics"] = diagnostics
        self.context["processing_info"] = {
            "strategies_used": len(all_candidates) if all_candidates else 0,
            "final_score": filtered_candidates[0][0] if filtered_candidates else 0.0,
            "scale_factor": scale
        }

        print("İşlem tamamlandı.")
        return build_response(context=self)


# -----------------------------------------------------------------------------
# 8. ÇALIŞTIRICI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    Executor(sys.argv[1]).run()