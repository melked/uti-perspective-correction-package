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


# ---------------------- Preprocessing Pipelines ---------------------- #

def _pre_soft(gray: np.ndarray) -> np.ndarray:
    """Desenli/karmaşık arka planlarda iyi: adaptif eşik + morfoloji."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 15, 11)
    th = cv2.medianBlur(th, 5)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    return th


def _pre_medium(gray: np.ndarray) -> np.ndarray:
    """Genel amaçlı: CLAHE + Canny."""
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    edges = cv2.Canny(eq, 50, 150, L2gradient=True)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    return edges


def _pre_hard(gray: np.ndarray) -> np.ndarray:
    """Uzak/az kontrast: unsharp + düşük eşikli Canny."""
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    unsharp = cv2.addWeighted(gray, 1.7, blur, -0.7, 0)
    edges = cv2.Canny(unsharp, 30, 100, L2gradient=True)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    return edges


def _pre_color_suppressed(gray: np.ndarray, bgr: np.ndarray) -> np.ndarray:
    """Renk bilgisini bastırarak kenarları ortaya çıkaran güçlü pipeline."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    L_eq = clahe.apply(L)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    grad = cv2.morphologyEx(L_eq, cv2.MORPH_GRADIENT, kernel)

    blur = cv2.GaussianBlur(grad, (0, 0), 2)
    sharp = cv2.addWeighted(grad, 1.5, blur, -0.5, 0)

    edges = cv2.Canny(sharp, 40, 120, L2gradient=True)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    return edges


def _generate_edge_maps(bgr: np.ndarray) -> List[np.ndarray]:
    """Tüm ön işleme stratejilerini uygulayarak kenar haritaları listesi oluşturur."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return [
        _pre_soft(gray_normalized),
        _pre_medium(gray_normalized),
        _pre_hard(gray_normalized),
        _pre_color_suppressed(gray_normalized, bgr)
    ]


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
    """Verilen 4 noktaya göre görüntüye perspektif dönüşümü uygular."""
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(round(max(widthA, widthB)))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(round(max(heightA, heightB)))

    maxWidth = max(2, maxWidth)
    maxHeight = max(2, maxHeight)

    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LANCZOS4)
    return warped


def _full_image_quad(image: np.ndarray) -> np.ndarray:
    """Varsayılan olarak tüm görüntüyü temsil eden dörtgeni döndürür."""
    h, w = image.shape[:2]
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)


def _min_area_rect_quad(binary_or_gray: np.ndarray, ref_image: np.ndarray) -> np.ndarray:
    """Son çare olarak, en büyük konturun etrafına minimum alanlı bir dörtgen çizer."""
    gray = binary_or_gray if len(binary_or_gray.shape) == 2 else cv2.cvtColor(binary_or_gray, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _full_image_quad(ref_image)
    c = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect).astype(np.float32)
    return box


# ---------------------- Candidate Generation ---------------------- #

def _find_quads_from_contours(binary_img: np.ndarray, ref_image: np.ndarray,
                              min_area_ratio: float = 0.02, max_area_ratio: float = 0.95) -> List[np.ndarray]:
    """İkili görüntüdeki konturları analiz ederek dörtgen adayları bulur."""
    if binary_img is None or binary_img.size == 0: return []
    binary_img = cv2.normalize(binary_img, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8) if binary_img.dtype != np.uint8 else binary_img

    contours, _ = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return []

    H, W = ref_image.shape[:2]
    img_area = H * W
    min_area = img_area * min_area_ratio
    max_area = img_area * max_area_ratio

    candidates = []
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(c)
        if not (min_area < area < max_area): continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            candidates.append(approx.reshape(4, 2).astype(np.float32))
    return candidates


def _find_quads_from_hough(edges: np.ndarray, ref_image: np.ndarray) -> List[np.ndarray]:
    """Hough çizgilerini kullanarak, özellikle açılı çekimlerde dörtgen adayları bulur."""
    edges_gray = edges if len(edges.shape) == 2 else cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
    lines = cv2.HoughLinesP(edges_gray, 1, np.pi / 180, threshold=60,
                            minLineLength=max(30, int(0.05 * min(edges.shape[:2]))),
                            maxLineGap=10)
    if lines is None: return []

    def angle_of(l):
        x1, y1, x2, y2 = l.ravel();
        return np.degrees(np.arctan2((y2 - y1), (x2 - x1) + 1e-6))

    group1, group2 = [], []
    for l in lines:
        a = (angle_of(l) + 180) % 180
        if 20 <= a <= 70 or 110 <= a <= 160:
            group1.append(l)
        else:
            group2.append(l)

    if len(group1) < 2 or len(group2) < 2: return []

    def top2_by_len(L):
        return sorted(L, key=lambda li: np.linalg.norm(li.ravel()[:2] - li.ravel()[2:]), reverse=True)[:2]

    g1, g2 = top2_by_len(group1), top2_by_len(group2)
    if len(g1) < 2 or len(g2) < 2: return []

    def line_to_abcd(l):
        x1, y1, x2, y2 = l.ravel();
        return (y1 - y2), (x2 - x1), (x1 * y2 - x2 * y1)

    def intersect(l1, l2):
        A1, B1, C1 = line_to_abcd(l1);
        A2, B2, C2 = line_to_abcd(l2)
        M = np.array([[A1, B1], [A2, B2]], dtype=np.float64)
        v = np.array([-C1, -C2], dtype=np.float64)
        if abs(np.linalg.det(M)) < 1e-6: return None
        try:
            x, y = np.linalg.solve(M, v); return float(x), float(y)
        except np.linalg.LinAlgError:
            return None

    pts = [p for l1 in g1 for l2 in g2 if (p := intersect(l1, l2)) is not None]

    if len(pts) < 4: return []
    pts = np.array(pts, dtype=np.float32)
    H, W = ref_image.shape[:2]
    pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H)]

    if pts.shape[0] < 4: return []

    hull = cv2.convexHull(pts)
    approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
    if len(approx) < 4: return []
    if len(approx) > 4:
        best, best_area = None, 0
        for comb in combinations(approx.reshape(-1, 2), 4):
            quad = np.array(comb, dtype=np.float32)
            area = cv2.contourArea(quad)
            if area > best_area: best_area, best = area, quad
        if best is None: return []
        quad = best
    else:
        quad = approx.reshape(4, 2).astype(np.float32)

    return [quad]


# ---------------------- Scoring ---------------------- #

def _angle_score(quad: np.ndarray) -> float:
    def angle(pt1, pt2, pt3):
        v1, v2 = pt1 - pt2, pt3 - pt2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    q = quad.astype(np.float32)
    angs = [angle(q[i], q[(i + 1) % 4], q[(i + 2) % 4]) for i in range(4)]
    return float(np.mean([1 - abs(a - 90) / 90 for a in angs]))


def _convexity_score(quad: np.ndarray) -> float:
    return 1.0 if cv2.isContourConvex(quad.astype(np.float32)) else 0.0


def _area_score(quad: np.ndarray, image: np.ndarray) -> float:
    area = abs(cv2.contourArea(quad.astype(np.float32)))
    H, W = image.shape[:2]
    return float(np.clip(area / (float(H * W) * 0.5), 0, 1))


def _shape_score(quad: np.ndarray) -> float:
    d = [np.linalg.norm(quad[i] - quad[(i + 1) % 4]) for i in range(4)]
    return float(1.0 - (np.std(d) / (np.mean(d) + 1e-6)))


def _aspect_ratio_score(quad: np.ndarray) -> float:
    rect = _order_points(quad)
    w = (np.linalg.norm(rect[0] - rect[1]) + np.linalg.norm(rect[2] - rect[3])) / 2
    h = (np.linalg.norm(rect[1] - rect[2]) + np.linalg.norm(rect[0] - rect[3])) / 2
    if min(w, h) < 1e-6: return 0.0
    ratio = max(w, h) / min(w, h)
    return float(np.clip(1.5 - abs(ratio - 1.5), 0, 1))


def _edge_density_score(quad: np.ndarray, edges_like: np.ndarray) -> float:
    gray = edges_like if len(edges_like.shape) == 2 else cv2.cvtColor(edges_like, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [quad.astype(np.int32)], 255)
    inside = cv2.countNonZero(cv2.bitwise_and(gray, gray, mask=mask))
    total = cv2.countNonZero(gray)
    if total == 0: return 0.0
    return float(np.clip((inside / total - 0.2) / 0.6, 0, 1))


def _parallelism_score(quad: np.ndarray) -> float:
    def seg_angle(p1, p2): v = p2 - p1; return np.degrees(np.arctan2(v[1], v[0] + 1e-6)) % 180

    q = _order_points(quad.copy())
    a1, a3 = seg_angle(q[0], q[1]), seg_angle(q[3], q[2])
    a2, a4 = seg_angle(q[1], q[2]), seg_angle(q[0], q[3])

    def closeness(a, b): d = min(abs(a - b), 180 - abs(a - b)); return 1 - d / 90

    return float(max(0.0, (closeness(a1, a3) + closeness(a2, a4)) / 2.0))


def _score_quad(quad: np.ndarray, image: np.ndarray, edges_for_density: np.ndarray) -> float:
    return (0.22 * _area_score(quad, image) +
            0.16 * _convexity_score(quad) +
            0.18 * _angle_score(quad) +
            0.12 * _shape_score(quad) +
            0.12 * _aspect_ratio_score(quad) +
            0.10 * _edge_density_score(quad, edges_for_density) +
            0.10 * _parallelism_score(quad))


# ---------------------- Selection ---------------------- #

def _select_best_quad(image: np.ndarray, edge_maps: List[np.ndarray],
                      contour_candidates_per_map: List[List[np.ndarray]],
                      hough_candidates: List[np.ndarray]) -> np.ndarray:
    scored: List[Tuple[np.ndarray, float]] = []

    for edges, candidates in zip(edge_maps, contour_candidates_per_map):
        for q in candidates:
            scored.append((q, _score_quad(q, image, edges)))

    if edge_maps:
        mid_edges = edge_maps[len(edge_maps) // 2]
        for q in hough_candidates:
            scored.append((q, _score_quad(q, image, mid_edges)))

    if scored:
        return max(scored, key=lambda x: x[1])[0]

    base = edge_maps[0] if edge_maps else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return _min_area_rect_quad(base, image)


# ---------------------- Component ---------------------- #

class PerspectiveCorrection(Component):
    """
    - Her arka plan için sağlam; uzaktan/çapraz çekimde dayanıklı.
    - Warp sonrası ek iyileştirme YOK (orijinal korunur).
    """

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
        if img is None or img.size == 0:
            raise ValueError("Input image empty or None")
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
            raise ValueError("No input image provided or failed to load")
        src = self._prepare_image(img_obj.value)

        edge_maps = _generate_edge_maps(src)

        contour_candidates_per_map = [_find_quads_from_contours(emap, src) for emap in edge_maps]
        hough_candidates = [h for emap in edge_maps for h in _find_quads_from_hough(emap, src)]

        best_quad = _select_best_quad(src, edge_maps, contour_candidates_per_map, hough_candidates)

        warped = _four_point_transform(src, best_quad)

        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)
        self.context["src_quad"] = np.array(best_quad, dtype=float).tolist()
        self.context["output_size"] = [int(warped.shape[1]), int(warped.shape[0])]
        self.context["method"] = {
            "pipelines": ["soft_adaptive", "medium_clahe_canny", "hard_unsharp_canny", "color_suppressed_lab"],
            "fallbacks": ["hough_lines", "min_area_rect", "full_image_if_needed"],
        }
        return build_response(context=self)


Executor(sys.argv[1]).run()