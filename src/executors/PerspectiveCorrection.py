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
    # Bu fonksiyon orijinal kodunuzda vardı, koruyoruz.
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

# YENİ -> Strateji 1 için
def _create_document_mask(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([0, 0, 120])
    upper_bound = np.array([180, 70, 255])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    return mask


# Orijinal pipeline fonksiyonlarınız
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


# GÜNCELLENDİ -> Strateji 2 için
def _generate_edge_maps(image_bgr: np.ndarray) -> List[np.ndarray]:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, _, _ = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    clahe_l = clahe.apply(l_channel)
    return [_pre_soft(clahe_l), _pre_medium(clahe_l), _pre_hard(clahe_l), _pre_extreme(clahe_l)]


# ---------------------- Candidate Generation (Orijinal) ---------------------- #

def _find_quads_from_contours(binary_img: np.ndarray, ref_image: np.ndarray) -> List[np.ndarray]:
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return []
    H, W = ref_image.shape[:2]
    min_area = H * W * 0.10
    candidates = []
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        if cv2.contourArea(c) < min_area: continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            candidates.append(approx.reshape(4, 2).astype(np.float32))
    return candidates


def _find_quads_from_hough(edges: np.ndarray, ref_image: np.ndarray) -> List[np.ndarray]:
    # Orijinal Hough fonksiyonunuz geri eklendi
    if len(edges.shape) == 3: edges = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                            minLineLength=max(30, int(0.05 * min(edges.shape[:2]))), maxLineGap=10)
    if lines is None: return []

    def line_to_abcd(l):
        x1, y1, x2, y2 = l.ravel();
        A = y1 - y2;
        B = x2 - x1;
        C = x1 * y2 - x2 * y1
        return A, B, C

    def intersect(l1, l2) -> Optional[Tuple[float, float]]:
        A1, B1, C1 = map(float, line_to_abcd(l1));
        A2, B2, C2 = map(float, line_to_abcd(l2))
        M = np.array([[A1, B1], [A2, B2]], dtype=np.float64);
        b = -np.array([C1, C2], dtype=np.float64)
        try:
            x, y = np.linalg.solve(M, b);
            return float(x), float(y)
        except np.linalg.LinAlgError:
            return None

    pts = [p for l1, l2 in combinations(lines, 2) if (p := intersect(l1, l2)) is not None]
    if len(pts) < 4: return []

    pts = np.array(pts, dtype=np.float32)
    H, W = ref_image.shape[:2]
    pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H)]
    if pts.shape[0] < 4: return []

    hull = cv2.convexHull(pts)
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
    if len(approx) == 4 and cv2.isContourConvex(approx):
        return [approx.reshape(4, 2).astype(np.float32)]
    return []


# ---------------------- Scoring (Orijinal) ---------------------- #
# Tüm detaylı skorlama fonksiyonlarınız korundu.

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
    H, W = image.shape[:2]
    ratio = area / float(H * W + 1e-6)
    return float(np.clip(ratio / 0.5, 0, 1))  # 0.5'i 1.0 kabul eder


def _shape_score(quad: np.ndarray) -> float:
    d = [np.linalg.norm(quad[i] - quad[(i + 1) % 4]) for i in range(4)]
    return float(1.0 - (np.std(d) / (np.mean(d) + 1e-6)))


def _aspect_ratio_score(quad: np.ndarray) -> float:
    rect = _order_points(quad)
    (tl, tr, br, bl) = rect
    w1 = np.linalg.norm(tr - tl)
    w2 = np.linalg.norm(br - bl)
    h1 = np.linalg.norm(bl - tl)
    h2 = np.linalg.norm(br - tr)
    w = max(w1, w2)
    h = max(h1, h2)
    if min(w, h) < 1e-6: return 0.0
    ratio = max(w, h) / min(w, h)
    # A4 kağıt oranı (yaklaşık 1.414) gibi belirli oranlara yakınlığı ödüllendirebilir
    return max(0, 1 - abs(ratio - 1.4) / 1.4)


def _edge_density_score(quad: np.ndarray, edges_like: np.ndarray) -> float:
    mask = np.zeros(edges_like.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [quad.astype(np.int32)], 255)
    inside_pixels = cv2.countNonZero(mask)
    if inside_pixels == 0: return 0.0

    edges_inside = cv2.countNonZero(cv2.bitwise_and(edges_like, edges_like, mask=mask))
    density = edges_inside / inside_pixels
    return float(np.clip(density, 0, 1.0))


def _parallelism_score(quad: np.ndarray) -> float:
    q = _order_points(quad.copy())

    def seg_angle(p1, p2): return np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))

    a_top = seg_angle(q[0], q[1]);
    a_bot = seg_angle(q[3], q[2])
    a_lft = seg_angle(q[0], q[3]);
    a_rgt = seg_angle(q[1], q[2])

    d_horiz = abs(a_top - a_bot);
    d_vert = abs(a_lft - a_rgt)
    return max(0, 1 - min(d_horiz, 180 - d_horiz) / 90) * max(0, 1 - min(d_vert, 180 - d_vert) / 90)


def _score_quad(quad: np.ndarray, image: np.ndarray, edges_for_density: np.ndarray) -> float:
    # Orijinal ağırlıklarınızla skorlama fonksiyonu
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
            s = _score_quad(q, image, edges)
            scored.append((q, s))
    if hough_candidates:
        for q in hough_candidates:
            s = _score_quad(q, image, edge_maps[-1])  # Son haritayı referans al
            scored.append((q, s))
    if not scored:
        return _full_image_quad(image)
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0]


# ---------------------- Main Component (Hibrit Strateji ile) ---------------------- #

class PerspectiveCorrection(Component):

    def __init__(self, config: PackageModel):
        super().__init__(config)

    @staticmethod
    def bootstrap(config: PackageModel):

        return {}

    def process(self, input_image: Image) -> Image:
        img_bgr = input_image.to_bgr()

        # --- Strateji 1: Renk Maskesi ---
        mask = _create_document_mask(img_bgr)
        candidates_from_mask = _find_quads_from_contours(mask, img_bgr)

        if candidates_from_mask:
            # Maskeden gelen adayları, maskenin kendisini yoğunluk haritası olarak kullanarak puanla
            scored_candidates = [(q, _score_quad(q, img_bgr, mask)) for q in candidates_from_mask]
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            best_candidate, best_score = scored_candidates[0]

            # Belirli bir skor eşiğini geçerse, bu sonucu kabul et
            if best_score > 0.4:  # Bu eşik ayarlanabilir
                warped = _four_point_transform(img_bgr, best_candidate)
                return Image.from_bgr(warped)

        # --- Strateji 2: LAB Renk Uzayı + Orijinal Pipeline (Yedek Plan) ---
        edge_maps = _generate_edge_maps(img_bgr)  # Bu fonksiyon artık LAB tabanlı çalışıyor
        contour_candidates = [_find_quads_from_contours(e, img_bgr) for e in edge_maps]
        hough_candidates = []
        for e in edge_maps:
            hough_candidates.extend(_find_quads_from_hough(e, img_bgr))

        best_quad = _select_best_quad(img_bgr, edge_maps, contour_candidates, hough_candidates)
        warped = _four_point_transform(img_bgr, best_quad)
        return Image.from_bgr(warped)


# ---------------------- Executor (Orijinal) ---------------------- #

if __name__ == "__main__":
    Executor(sys.argv[1]).run()