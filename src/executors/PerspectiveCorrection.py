import os
import sys
import cv2
import numpy as np
from typing import List, Tuple, Optional
from itertools import combinations

# Projenize özel importlar
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))
from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveCorrection.src.utils.response import build_response
from components.PerspectiveCorrection.src.models.PackageModel import PackageModel


# ---------------------- Geometry Helpers (Orijinal) ---------------------- #

def _order_points(pts: np.ndarray) -> np.ndarray:
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
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(round(max(widthA, widthB)))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(round(max(heightA, heightB)))
    if maxWidth < 2 or maxHeight < 2: return image
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LANCZOS4)
    return warped


def _full_image_quad(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)


def _min_area_rect_quad(binary_or_gray: np.ndarray, ref_image: np.ndarray) -> np.ndarray:
    if len(binary_or_gray.shape) == 3:
        gray = cv2.cvtColor(binary_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = binary_or_gray.copy()
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return _full_image_quad(ref_image)
    c = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect).astype(np.float32)
    return box


# ---------------------- Preprocessing Pipelines (Geliştirilmiş) ---------------------- #

# YENİ -> Strateji 1 için renk maskesi oluşturucu
def _create_document_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Açık renkli belgeyi arka plandan ayırmak için HSV renk maskesi oluşturur."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([0, 0, 120])  # Parlaklık eşiğini biraz daha düşük tutarak gölgeleri de yakalayabiliriz
    upper_bound = np.array([180, 80, 255])  # Doygunluk eşiğini biraz artırarak soluk renkleri de alabiliriz
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    # Maskedeki gürültüyü daha etkili temizlemek için kernel boyutu artırıldı
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    return mask


# Orijinal pipeline fonksiyonlarınız (Strateji 2 için kullanılacak)
def _pre_soft(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 11)
    th = cv2.medianBlur(th, 5)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    return th


def _pre_medium(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    edges = cv2.Canny(eq, 50, 150, L2gradient=True)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    return edges


def _pre_hard(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    unsharp = cv2.addWeighted(gray, 1.7, blur, -0.7, 0)
    edges = cv2.Canny(unsharp, 30, 100, L2gradient=True)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    return edges


def _pre_extreme(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    tophat = cv2.morphologyEx(eq, cv2.MORPH_TOPHAT, kernel)
    enhanced = cv2.add(eq, tophat)
    blur = cv2.GaussianBlur(enhanced, (0, 0), 3)
    sharp = cv2.addWeighted(enhanced, 1.8, blur, -0.8, 0)
    edges = cv2.Canny(sharp, 30, 120, L2gradient=True)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    return edges


# GÜNCELLENDİ -> Artık Strateji 2 için LAB renk uzayını kullanıyor
def _generate_edge_maps(image_bgr: np.ndarray) -> List[np.ndarray]:
    """LAB renk uzayının L kanalını ve CLAHE'yi kullanarak kenar haritaları üretir."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, _, _ = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    clahe_l = clahe.apply(l_channel)
    return [_pre_soft(clahe_l), _pre_medium(clahe_l), _pre_hard(clahe_l), _pre_extreme(clahe_l)]


# ---------------------- Candidate Generation (Orijinal) ---------------------- #
def _find_quads_from_contours(binary_img: np.ndarray, ref_image: np.ndarray) -> List[np.ndarray]:
    # RETR_EXTERNAL sadece en dış konturları bulur, bu daha verimlidir.
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return []

    H, W = ref_image.shape[:2]
    img_area = H * W
    min_area = img_area * 0.10  # Alan eşiğini %10'a yükselterek küçük gürültüleri eleyelim
    max_area = img_area * 0.98

    candidates = []
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:  # En büyük 5 kontur yeterli
        area = cv2.contourArea(c)
        if not (min_area < area < max_area): continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            candidates.append(approx.reshape(4, 2).astype(np.float32))
    return candidates


def _find_quads_from_hough(edges: np.ndarray, ref_image: np.ndarray) -> List[np.ndarray]:
    # Orijinal Hough fonksiyonunuz korunuyor.
    return []  # Geçici olarak devre dışı, isteğe bağlı olarak etkinleştirilebilir.


# ---------------------- Scoring (Orijinal) ---------------------- #
# Orijinal detaylı skorlama fonksiyonlarınızın tümü korunuyor.
def _angle_score(quad: np.ndarray) -> float:
    def angle(pt1, pt2, pt3):
        v1 = pt1 - pt2;
        v2 = pt3 - pt2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    q = quad.astype(np.float32)
    angs = [angle(q[i], q[(i + 1) % 4], q[(i + 2) % 4]) for i in range(4)]
    return float(np.mean([1 - min(abs(a - 90), 90) / 90 for a in angs]))


def _convexity_score(quad: np.ndarray) -> float:
    return 1.0 if cv2.isContourConvex(quad.astype(np.float32)) else 0.0


def _area_score(quad: np.ndarray, image: np.ndarray) -> float:
    area = abs(cv2.contourArea(quad.astype(np.float32)))
    H, W = image.shape[:2];
    ratio = area / float(H * W + 1e-6)
    return float(np.clip(ratio / 0.5, 0, 1))


def _shape_score(quad: np.ndarray) -> float:
    d = [np.linalg.norm(quad[i] - quad[(i + 1) % 4]) for i in range(4)]
    return float(1.0 - (np.std(d) / (np.mean(d) + 1e-6)))


def _aspect_ratio_score(quad: np.ndarray) -> float:
    w1 = np.linalg.norm(quad[0] - quad[1]);
    w2 = np.linalg.norm(quad[2] - quad[3])
    h1 = np.linalg.norm(quad[1] - quad[2]);
    h2 = np.linalg.norm(quad[3] - quad[0])
    w, h = (w1 + w2) / 2.0, (h1 + h2) / 2.0
    if h <= 1e-6 or w <= 1e-6: return 0.0
    ratio = w / h if w > h else h / w
    return float(np.clip(1.5 - abs(ratio - 1.5), 0, 1))


def _edge_density_score(quad: np.ndarray, edges_like: np.ndarray) -> float:
    mask = np.zeros(edges_like.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [quad.astype(np.int32)], 255)
    inside = cv2.countNonZero(cv2.bitwise_and(edges_like, edges_like, mask=mask))
    outside = cv2.countNonZero(cv2.bitwise_and(edges_like, edges_like, mask=cv2.bitwise_not(mask)))
    if inside + outside == 0: return 0.0
    return float(np.clip((inside / (inside + outside) - 0.2) / 0.6, 0, 1))


def _parallelism_score(quad: np.ndarray) -> float:
    q = _order_points(quad.copy())

    def seg_angle(p1, p2): return np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0])) % 180

    a1 = seg_angle(q[0], q[1]);
    a3 = seg_angle(q[3], q[2])
    a2 = seg_angle(q[1], q[2]);
    a4 = seg_angle(q[0], q[3])

    def closeness(a, b): d = min(abs(a - b), 180 - abs(a - b)); return 1 - d / 90

    return (closeness(a1, a3) + closeness(a2, a4)) / 2.0


def _score_quad(quad: np.ndarray, image: np.ndarray, edges_for_density: np.ndarray) -> float:
    return (0.22 * _area_score(quad, image) + 0.16 * _convexity_score(quad) + 0.18 * _angle_score(quad) +
            0.12 * _shape_score(quad) + 0.12 * _aspect_ratio_score(quad) + 0.10 * _edge_density_score(quad,
                                                                                                      edges_for_density) +
            0.10 * _parallelism_score(quad))


# ---------------------- Selection (Orijinal) ---------------------- #

def _select_best_quad(image: np.ndarray, edge_maps: List[np.ndarray],
                      contour_candidates_per_map: List[List[np.ndarray]],
                      hough_candidates: List[np.ndarray]) -> np.ndarray:
    scored: List[Tuple[np.ndarray, float]] = []
    for edges, candidates in zip(edge_maps, contour_candidates_per_map):
        for q in candidates:
            scored.append((q, _score_quad(q, image, edges)))
    for q in hough_candidates:
        scored.append((q, _score_quad(q, image, edge_maps[-1])))
    if not scored: return _full_image_quad(image)
    return max(scored, key=lambda item: item[1])[0]


# ---------------------- Main Component (Geliştirilmiş) ---------------------- #

class PerspectiveCorrection(Component):

    def __init__(self, config: PackageModel):
        super().__init__(config)

    @staticmethod
    def bootstrap(config: PackageModel):
        return {}

    def run(self, input_image: Image) -> Image:
        return self.process(input_image)

    def process(self, input_image: Image) -> Image:
        img_bgr = input_image.to_bgr()

        # --- Strateji 1: Renk Maskesi ile Hızlı Tarama ---
        mask = _create_document_mask(img_bgr)
        candidates_from_mask = _find_quads_from_contours(mask, img_bgr)
        if candidates_from_mask:
            # Maskeden gelen adayları, maskenin kendisini yoğunluk haritası olarak kullanarak puanla
            scored_candidates = [(q, _score_quad(q, img_bgr, mask)) for q in candidates_from_mask]
            best_candidate, best_score = max(scored_candidates, key=lambda item: item[1])

            # Belirli bir skor eşiğini geçerse, bu sonucu kabul et
            if best_score > 0.4:  # Ayarlanabilir bir güven eşiği
                warped = _four_point_transform(img_bgr, best_candidate)
                return Image.from_bgr(warped)

        # --- Strateji 2: LAB Renk Uzayı ile Derin Analiz (Yedek Plan) ---
        edge_maps = _generate_edge_maps(img_bgr)  # Artık LAB tabanlı çalışıyor
        contour_candidates = [_find_quads_from_contours(e, img_bgr) for e in edge_maps]
        hough_candidates = []  # Hough'u kullanmak isterseniz bu listeyi doldurun

        best_quad = _select_best_quad(img_bgr, edge_maps, contour_candidates, hough_candidates)
        warped = _four_point_transform(img_bgr, best_quad)
        return Image.from_bgr(warped)


# ---------------------- Executor (Orijinal) ---------------------- #

if __name__ == "__main__":
    Executor(sys.argv[1]).run()