import os
import sys
import cv2
import numpy as np
import math
from collections import defaultdict
from typing import Optional, List, Tuple
from itertools import combinations

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
    """ Tüm uzmanların ve yardımcı fonksiyonların kullandığı parametreleri merkezi olarak yönetir. """

    def __init__(self, config=None):
        config = config or {}
        # Genel Puanlama
        self.score_min_area_ratio = config.get("score_min_area_ratio", 0.03)
        self.score_max_area_ratio = config.get("score_max_area_ratio", 0.95)
        self.score_min_threshold = config.get("score_min_threshold", 0.25)

        # Stage 1: Hızlı Gözcü
        self.s1_blur_ksize = tuple(config.get("s1_blur_ksize", (5, 5)))

        # Stage 2: Sınır Gözcüsü
        self.s2_blur_ratio = config.get("s2_blur_ratio", 5)
        self.s2_morph_ksize = tuple(config.get("s2_morph_ksize", (7, 7)))
        self.s2_morph_iter = config.get("s2_morph_iter", 3)

        # Stage 3: Noktasal Köşe Avcısı
        self.s3_max_corners = config.get("s3_max_corners", 100)
        self.s3_quality_level = config.get("s3_quality_level", 0.01)
        self.s3_min_distance = config.get("s3_min_distance", 20)

        # Stage 4: İçerik Analisti
        self.s4_resize_longest_edge = config.get("s4_resize_longest_edge", 300)
        self.s4_kmeans_clusters = config.get("s4_kmeans_clusters", 3)

        # Stage 5: Çizgi Dedektifi
        self.s5_hough_threshold_ratio = config.get("s5_hough_threshold_ratio", 4)


# -----------------------------------------------------------------------------
# 2. Geometri, Puanlama ve Yardımcı Fonksiyonlar
# -----------------------------------------------------------------------------
def _order_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1);
    rect[0] = pts[np.argmin(s)];
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1);
    rect[1] = pts[np.argmin(diff)];
    rect[3] = pts[np.argmax(diff)]
    return rect


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
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


def _score_candidate(contour: np.ndarray, params: Params, image_shape: tuple) -> float:
    h, w = image_shape[:2];
    total_area = w * h
    area = cv2.contourArea(contour)
    if not (params.score_min_area_ratio < area / total_area < params.score_max_area_ratio): return 0.0
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    if len(approx) != 4 or not cv2.isContourConvex(approx): return 0.0
    return area / total_area


def _get_canny_param_sets(image: np.ndarray) -> List[Tuple[int, int]]:
    v = np.median(image);
    sigma = 0.33
    auto_lower = int(max(0, (1.0 - sigma) * v));
    auto_upper = int(min(255, (1.0 + sigma) * v))
    return list(dict.fromkeys([(auto_lower, auto_upper), (30, 90), (50, 150)]))


def _line_intersection(line1, line2):
    rho1, theta1 = line1;
    rho2, theta2 = line2
    A = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
    b = np.array([[rho1], [rho2]])
    try:
        x0, y0 = np.linalg.solve(A, b)
        return [int(round(x0[0])), int(round(y0[0]))]
    except np.linalg.LinAlgError:
        return None


# -----------------------------------------------------------------------------
# 3. UZMAN STRATEJİLERİ
# -----------------------------------------------------------------------------

def stage1_fast_and_simple(image: np.ndarray, params: Params) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, params.s1_blur_ksize, 0)
    best_quad, best_score = None, params.score_min_threshold
    for lower, upper in _get_canny_param_sets(blurred):
        edged = cv2.Canny(blurred, lower, upper)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue
        c = max(contours, key=cv2.contourArea)
        score = _score_candidate(c, params, image.shape)
        if score > best_score:
            best_score = score
            peri = cv2.arcLength(c, True)
            best_quad = cv2.approxPolyDP(c, params.approx_poly_epsilon_ratio * peri, True)
    return best_quad.reshape(4, 2).astype(np.float32) if best_quad is not None else None


def stage2_boundary_watcher(image: np.ndarray, params: Params) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel_size = int(min(image.shape[:2]) / params.s2_blur_ratio)
    if kernel_size % 2 == 0: kernel_size += 1
    blurred_bg = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    flattened = cv2.divide(gray, blurred_bg, scale=255)
    _, thresh = cv2.threshold(flattened, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, params.s2_morph_ksize)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=params.s2_morph_iter)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    c = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, params.approx_poly_epsilon_ratio * peri, True)
    if len(approx) == 4 and _score_candidate(c, params, image.shape) > params.score_min_threshold:
        return approx.reshape(4, 2).astype(np.float32)
    return None


def stage3_feature_detector(image: np.ndarray, params: Params) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=params.s3_max_corners, qualityLevel=params.s3_quality_level,
                                      minDistance=params.s3_min_distance)
    if corners is None or len(corners) < 4: return None
    hull = cv2.convexHull(corners)
    if _score_candidate(hull, params, image.shape) > params.score_min_threshold:
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, params.approx_poly_epsilon_ratio * peri, True)
        if len(approx) == 4:
            return _order_points(approx.reshape(4, 2)).astype(np.float32)
    return None


def stage4_content_analyzer(image: np.ndarray, params: Params) -> Optional[np.ndarray]:
    h, w = image.shape[:2];
    total_area = h * w
    scale = params.s4_resize_longest_edge / max(h, w)
    small_img = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    pixels = small_img.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, params.s4_kmeans_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = centers.astype(np.uint8)
    lab_centers = cv2.cvtColor(centers.reshape(1, -1, 3), cv2.COLOR_BGR2LAB)[0]
    brightest_idx = np.argmax([c[0] for c in lab_centers])
    mask = (labels.reshape(small_img.shape[:2]) == brightest_idx).astype(np.uint8) * 255
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    c = max(contours, key=cv2.contourArea)
    if not (params.score_min_area_ratio < cv2.contourArea(c) / total_area < params.score_max_area_ratio): return None
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    return box.astype(np.float32)


def stage5_line_reconstructor(image: np.ndarray, params: Params) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), params.canny_min, params.canny_max)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, int(min(image.shape[:2]) / params.s5_hough_threshold_ratio))
    if lines is None: return None
    h_lines, v_lines = [], []
    for line in lines:
        rho, theta = line[0]
        if theta < np.pi / 4 or theta > 3 * np.pi / 4:
            v_lines.append((rho, theta))
        else:
            h_lines.append((rho, theta))
    if len(h_lines) < 2 or len(v_lines) < 2: return None
    h_lines.sort(key=lambda x: x[0]);
    v_lines.sort(key=lambda x: x[0])
    corners = [_line_intersection(h_lines[0], v_lines[0]), _line_intersection(h_lines[0], v_lines[-1]),
               _line_intersection(h_lines[-1], v_lines[-1]), _line_intersection(h_lines[-1], v_lines[0])]
    if any(c is None for c in corners): return None
    quad = np.array(corners, dtype=np.float32)
    if _score_candidate(quad, params, image.shape) > 0.1:
        return quad
    return None


# -----------------------------------------------------------------------------
# Ana Bileşen
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
        src_img = self._prepare_image(img_obj.value)
        h, w = src_img.shape[:2]

        document_quad = None
        warped = None

        strategies = {
            "Hızlı Gözcü": stage1_fast_and_simple,
            "Sınır Gözcüsü": stage2_boundary_watcher,
            "Noktasal Köşe Avcısı": stage3_feature_detector,
            "İçerik Analisti": stage4_content_analyzer,
            "Çizgi Dedektifi": stage5_line_reconstructor,
        }

        for name, strategy in strategies.items():
            print(f"Aşama ( {name} ) deneniyor...")
            candidate_quad = strategy(src_img, self.params)
            if candidate_quad is not None:
                warped_candidate = _four_point_transform(src_img, candidate_quad)
                if warped_candidate is not None:
                    print(f"Başarılı: Belge '{name}' stratejisi ile bulundu ve doğrulandı.")
                    document_quad = candidate_quad
                    warped = warped_candidate
                    break

        if warped is None:
            print("Tüm uzmanlar başarısız. Fallback olarak tüm görüntü kullanılıyor.")
            warped = src_img
            document_quad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)
        self.context["src_quad"] = document_quad.tolist()
        self.context["output_size"] = [warped.shape[1], warped.shape[0]]
        return build_response(context=self)


# -----------------------------------------------------------------------------
# Çalıştırıcı
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    Executor(sys.argv[1]).run()