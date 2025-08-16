
import os
import sys
import cv2
import numpy as np
from typing import List, Tuple, Optional
from itertools import combinations

# SDK Entegrasyonu
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))
from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveCorrection.src.utils.response import build_response
from components.PerspectiveCorrection.src.models.PackageModel import PackageModel

# ---------------------- Merkezi Konfigürasyon ---------------------- #

CONFIG = {
    # Ön İşleme Parametreleri
    "gaussian_blur_kernel": (5, 5),
    "clahe_clip_limit": 2.5,
    "clahe_grid_size": (8, 8),
    "canny_threshold1": 50,
    "canny_threshold2": 150,
    "dilation_kernel_size": (3, 3),

    # Renk Tabanlı Maskeleme Parametreleri
    "background_saturation_threshold": 40,  # Doygunluk eşiği (renkli yüzeyler için)
    "background_value_threshold": 40,  # Parlaklık eşiği (koyu yüzeyler için)
    "background_morph_kernel_size": (9, 9),

    # Aday Tespiti Parametreleri
    "contour_min_area_ratio": 0.1,
    "approx_poly_epsilon": 0.02,
    "hough_threshold": 50,
    "hough_min_line_length": 50,
    "hough_max_line_gap": 15,

    # Seçim Parametreleri
    "unique_candidate_min_distance": 30.0,  # Benzersiz adaylar arası min piksel mesafesi
    "min_score_threshold": 0.2,  # Kabul edilebilir minimum aday skoru

    # Puanlama Ağırlıkları
    "score_weights": {
        "area": 0.30,
        "angle": 0.30,
        "parallelism": 0.25,
        "convexity": 0.10,
        "aspect_ratio": 0.05,
    },

    # Sağlamlık Parametreleri
    "min_warped_size": 50,  # Düzeltilmiş görüntünün minimum boyutu (px)
}


# ---------------------- Geometri Yardımcıları ---------------------- #

def _order_points(pts: np.ndarray) -> np.ndarray:
    """4 noktayı saat yönünde sol-üstten başlayarak sıralar."""
    pts = pts.reshape(4, 2).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Verilen 4 noktaya göre görüntüye perspektif dönüşümü uygular."""
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect
    width = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
    height = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))
    maxWidth, maxHeight = int(round(width)), int(round(height))

    if maxWidth < CONFIG["min_warped_size"] or maxHeight < CONFIG["min_warped_size"]:
        # Eğer sonuç çok küçükse, muhtemelen hatalı bir tespittir. Orijinali döndür.
        return image

    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LANCZOS4)


def _full_image_quad(image: np.ndarray) -> np.ndarray:
    """Hiç aday bulunamazsa tüm görüntüyü temsil eden dörtgeni döndürür."""
    h, w = image.shape[:2]
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)


def _min_area_rect_quad(binary_img: np.ndarray, ref_image: np.ndarray) -> np.ndarray:
    """Son çare olarak, ikili görüntüdeki en büyük nesnenin etrafına bir dörtgen çizer."""
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return _full_image_quad(ref_image)
    c = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    return cv2.boxPoints(rect).astype(np.float32)


# ---------------------- Ön İşleme Stratejileri (Pipelines) ---------------------- #

def _pre_soft(gray: np.ndarray) -> np.ndarray:
    """Karmaşık arka planlar için adaptif eşikleme."""
    blur = cv2.GaussianBlur(gray, CONFIG["gaussian_blur_kernel"], 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 11)
    return cv2.medianBlur(th, 5)


def _pre_medium(gray: np.ndarray) -> np.ndarray:
    """Genel amaçlı, kontrastı artırılmış kenar tespiti."""
    eq = cv2.createCLAHE(clipLimit=CONFIG["clahe_clip_limit"], tileGridSize=CONFIG["clahe_grid_size"]).apply(gray)
    edges = cv2.Canny(eq, CONFIG["canny_threshold1"], CONFIG["canny_threshold2"], L2gradient=True)
    return cv2.dilate(edges, np.ones(CONFIG["dilation_kernel_size"], np.uint8), iterations=1)


def _pre_hard(gray: np.ndarray) -> np.ndarray:
    """Düşük kontrastlı durumlar için keskinleştirilmiş kenar tespiti."""
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    unsharp = cv2.addWeighted(gray, 1.7, blur, -0.7, 0)
    edges = cv2.Canny(unsharp, 30, 100, L2gradient=True)
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones(CONFIG["dilation_kernel_size"], np.uint8), iterations=1)


def _create_background_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Doygun (renkli) ve/veya karanlık arka plan alanlarını tespit ederek bir maske oluşturur."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    _, sat_mask = cv2.threshold(saturation, CONFIG["background_saturation_threshold"], 255, cv2.THRESH_BINARY)
    _, val_mask = cv2.threshold(value, CONFIG["background_value_threshold"], 255, cv2.THRESH_BINARY_INV)
    mask = cv2.bitwise_or(sat_mask, val_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, CONFIG["background_morph_kernel_size"])
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)


def _pre_color_robust(image_bgr: np.ndarray) -> np.ndarray:
    """En güçlü pipeline: Renk maskeleme ile arka planı eleyerek kenar tespiti yapar."""
    background_mask = _create_background_mask(image_bgr)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    l_enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l_channel)
    l_enhanced[background_mask == 255] = 255
    blur = cv2.GaussianBlur(l_enhanced, CONFIG["gaussian_blur_kernel"], 0)
    edges = cv2.Canny(blur, CONFIG["canny_threshold1"], CONFIG["canny_threshold2"], L2gradient=True)
    return cv2.dilate(edges, np.ones(CONFIG["dilation_kernel_size"], np.uint8), iterations=1)


def _generate_edge_maps(image_bgr: np.ndarray) -> List[np.ndarray]:
    """Tüm ön işleme stratejilerini uygulayarak bir kenar haritaları listesi döndürür."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return [_pre_soft(gray), _pre_medium(gray), _pre_hard(gray), _pre_color_robust(image_bgr)]


# ---------------------- Aday Dörtgen Üretimi ---------------------- #

def _find_quads_from_contours(binary_img: np.ndarray, ref_image: np.ndarray) -> List[np.ndarray]:
    """İkili görüntüdeki konturları analiz ederek dörtgen adayları bulur."""
    contours, _ = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return []
    H, W = ref_image.shape[:2]
    min_area = H * W * CONFIG["contour_min_area_ratio"]
    candidates = []
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        if cv2.contourArea(c) < min_area: continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, CONFIG["approx_poly_epsilon"] * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            candidates.append(approx.reshape(4, 2).astype(np.float32))
    return candidates


def _find_quads_from_hough(edges: np.ndarray) -> List[np.ndarray]:
    """Hough çizgilerini kullanarak, özellikle açılı çekimlerde dörtgen adayları bulur."""
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, CONFIG["hough_threshold"],
                            minLineLength=CONFIG["hough_min_line_length"],
                            maxLineGap=CONFIG["hough_max_line_gap"])
    if lines is None: return []

    horizontal, vertical = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle < 30 or angle > 150:
            horizontal.append(line[0])
        elif 60 < angle < 120:
            vertical.append(line[0])

    if len(horizontal) < 2 or len(vertical) < 2: return []

    def line_intersect(l1, l2):
        x1, y1, x2, y2 = l1;
        x3, y3, x4, y4 = l2
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if den == 0: return None
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den
        if 0 < t < 1 and 0 < u < 1:
            return [int(x1 + t * (x2 - x1)), int(y1 + t * (y2 - y1))]
        return None

    quads = []
    for h1, h2 in combinations(horizontal, 2):
        for v1, v2 in combinations(vertical, 2):
            p = [line_intersect(h1, v1), line_intersect(h1, v2), line_intersect(h2, v2), line_intersect(h2, v1)]
            if all(pt is not None for pt in p):
                quads.append(np.array(p, dtype=np.float32))
    return quads


# ---------------------- Puanlama ve Seçim ---------------------- #

def _score_quad(quad: np.ndarray, image: np.ndarray) -> float:
    """Bir dörtgen adayını, merkezi konfigürasyondaki ağırlıklara göre puanlar."""
    area = abs(cv2.contourArea(quad))
    img_area = image.shape[0] * image.shape[1]

    if not cv2.isContourConvex(quad) or area < CONFIG["contour_min_area_ratio"] * img_area:
        return 0.0

    # Bireysel skorları hesapla
    area_score = np.clip(area / (0.95 * img_area), 0, 1)

    q_ord = _order_points(quad)

    def angle(p1, p2, p3):
        v1, v2 = p1 - p2, p3 - p2;
        return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6), -1, 1)))

    angles = [angle(q_ord[i], q_ord[(i + 1) % 4], q_ord[(i + 2) % 4]) for i in range(4)]
    angle_score = np.mean([1 - abs(a - 90) / 90 for a in angles])

    def seg_angle(p1, p2): return np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0] + 1e-6)) % 180

    a1, a3 = seg_angle(q_ord[0], q_ord[1]), seg_angle(q_ord[3], q_ord[2])
    a2, a4 = seg_angle(q_ord[1], q_ord[2]), seg_angle(q_ord[0], q_ord[3])

    def closeness(a, b): d = min(abs(a - b), 180 - abs(a - b)); return 1 - d / 30

    parallel_score = (closeness(a1, a3) + closeness(a2, a4)) / 2

    w = max(np.linalg.norm(q_ord[0] - q_ord[1]), np.linalg.norm(q_ord[2] - q_ord[3]))
    h = max(np.linalg.norm(q_ord[1] - q_ord[2]), np.linalg.norm(q_ord[3] - q_ord[0]))
    ar = min(w, h) / (max(w, h) + 1e-6)
    ar_score = 1 - abs(ar - 0.707)  # A4 (1/sqrt(2)) oranına yakınlık

    # Ağırlıklı toplamı hesapla
    w = CONFIG["score_weights"]
    final_score = (w["area"] * area_score + w["angle"] * angle_score +
                   w["parallelism"] * parallel_score + w["convexity"] * 1.0 +
                   w["aspect_ratio"] * ar_score)
    return final_score / sum(w.values())


def _gather_all_candidates(edge_maps: List[np.ndarray], image: np.ndarray) -> List[np.ndarray]:
    """Tüm aday üretme yöntemlerini kullanarak adayları toplar."""
    candidates = []
    for emap in edge_maps:
        candidates.extend(_find_quads_from_contours(emap, image))
        candidates.extend(_find_quads_from_hough(emap))
    return candidates


def _filter_unique_candidates(candidates: List[np.ndarray]) -> List[np.ndarray]:
    """Birbirine çok benzeyen aday dörtgenleri listeden çıkarır."""
    if not candidates: return []

    unique = []
    candidates.sort(key=cv2.contourArea, reverse=True)
    unique.append(candidates[0])

    for c in candidates[1:]:
        is_duplicate = any(
            np.linalg.norm(c.mean(axis=0) - u.mean(axis=0)) < CONFIG["unique_candidate_min_distance"] for u in unique)
        if not is_duplicate:
            unique.append(c)
    return unique


def _select_best_quad(image: np.ndarray, edge_maps: List[np.ndarray]) -> np.ndarray:
    """Tüm adayları toplar, filtreler, puanlar ve en iyisini seçer."""
    all_candidates = _gather_all_candidates(edge_maps, image)
    unique_candidates = _filter_unique_candidates(all_candidates)

    if not unique_candidates:
        # Son çare olarak en büyük nesneyi bul
        return _min_area_rect_quad(edge_maps[0], image)

    # Puanlama için en güvenilir kenar haritasını (renk tabanlı olanı) kullan
    scoring_edges = edge_maps[-1]
    scored_candidates = [(q, _score_quad(q, image, scoring_edges)) for q in unique_candidates]

    best_candidate, best_score = max(scored_candidates, key=lambda x: x[1])

    if best_score < CONFIG["min_score_threshold"]:
        return _min_area_rect_quad(edge_maps[0], image)

    return best_candidate


# ---------------------- Ana Component Sınıfı ---------------------- #

class PerspectiveCorrection(Component):
    """
    Görüntüdeki bir nesnenin perspektifini algılayan ve düzelten ana bileşen.
    """

    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.context = {}
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        """SDK için gerekli olan ve boş konfigürasyon döndüren başlangıç metodu."""
        return {}

    @staticmethod
    def _prepare_image(img: np.ndarray) -> np.ndarray:
        """Giriş görüntüsünü BGR formatında ve uint8 tipinde standart hale getirir."""
        if img is None or img.size == 0: raise ValueError("Giriş görüntüsü boş veya yüklenemedi.")
        if img.dtype != np.uint8: img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def run(self):
        """Component'in ana iş akışını çalıştırır."""
        # 1. Görüntüyü Hazırla
        img_obj = Image.get_frame(img=self.image, redis_db=self.redis_db)
        src = self._prepare_image(img_obj.value)

        # 2. Kenar Haritaları Üret
        edge_maps = _generate_edge_maps(src)

        # 3. En İyi Dörtgeni Seç
        best_quad = _select_best_quad(src, edge_maps)

        # 4. Perspektifi Düzelt
        warped = _four_point_transform(src, best_quad)

        # 5. Sonuçları Kaydet ve Raporla
        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)
        self.context["src_quad"] = best_quad.tolist()
        self.context["output_size"] = [warped.shape[1], warped.shape[0]]
        self.context["method"] = {
            "pipelines": ["soft", "medium", "hard", "color_robust"],
            "candidate_generation": ["contours", "hough_lines"],
            "selection_fallbacks": ["min_area_rect", "full_image"],
        }
        return build_response(context=self)


# ---------------------- Çalıştırıcı ---------------------- #

Executor(sys.argv[1]).run()