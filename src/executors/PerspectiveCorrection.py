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
    """ Tüm sistemin davranışını kontrol eden merkezi ayar noktası. """

    def __init__(self, config=None):
        config = config or {}
        # Genel
        self.resize_longest_edge = config.get("resize_longest_edge", 1000)
        self.unsharp_strength = config.get("unsharp_strength", 1.7)
        self.score_min_area_ratio = config.get("score_min_area_ratio", 0.04)
        self.score_max_area_ratio = config.get("score_max_area_ratio", 0.95)
        self.min_score_threshold = config.get("min_score_threshold", 0.25)
        self.approx_poly_epsilon_ratio = config.get("approx_poly_epsilon_ratio", 0.02)

        # Uzmanlar İçin Ayarlar
        self.canny_min = config.get("canny_min", 50)
        self.canny_max = config.get("canny_max", 150)
        self.s2_tophat_ksize = tuple(config.get("s2_tophat_ksize", (25, 25)))
        self.s3_blur_ratio = config.get("s3_blur_ratio", 5)
        self.s4_max_corners = config.get("s4_max_corners", 100)
        self.s4_quality_level = config.get("s4_quality_level", 0.01)
        self.s4_min_distance = config.get("s4_min_distance", 20)
        self.s5_kmeans_clusters = config.get("s5_kmeans_clusters", 3)
        self.s6_hough_threshold_ratio = config.get("s6_hough_threshold_ratio", 4)


# -----------------------------------------------------------------------------
# 2. Geometri, Puanlama ve Yardımcı Fonksiyonlar
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
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return sharpened


def _score_candidate(contour: np.ndarray, params: Params, image_shape: tuple) -> float:
    h, w = image_shape[:2];
    total_area = w * h
    area = cv2.contourArea(contour)
    if not (params.score_min_area_ratio < area / total_area < params.score_max_area_ratio): return 0.0
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, params.approx_poly_epsilon_ratio * peri, True)
    if len(approx) != 4 or not cv2.isContourConvex(approx): return 0.0
    return area / total_area


def _line_intersection(line1, line2):
    rho1, theta1 = line1;
    rho2, theta2 = line2
    A = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
    b = np.array([[rho1], [rho2]])
    try:
        return [int(round(c[0])) for c in np.linalg.solve(A, b)]
    except np.linalg.LinAlgError:
        return None


# -----------------------------------------------------------------------------
# 3. UZMAN STRATEJİLERİ
# -----------------------------------------------------------------------------

def stage1_fast_and_simple(image: np.ndarray, params: Params) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, params.canny_min, params.canny_max)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    c = max(contours, key=cv2.contourArea)
    if _score_candidate(c, params, image.shape) > params.min_score_threshold:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, params.approx_poly_epsilon_ratio * peri, True)
        return approx.reshape(4, 2).astype(np.float32)
    return None


def stage2_low_contrast_specialist(image: np.ndarray, params: Params) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, params.s2_tophat_ksize)
    tophat = cv2.morphologyEx(enhanced_gray, cv2.MORPH_TOPHAT, kernel)
    _, thresh = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    c = max(contours, key=cv2.contourArea)
    if _score_candidate(c, params, image.shape) > params.min_score_threshold:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, params.approx_poly_epsilon_ratio * peri, True)
        return approx.reshape(4, 2).astype(np.float32)
    return None


def stage3_boundary_watcher(image: np.ndarray, params: Params) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel_size = int(min(image.shape[:2]) / params.s3_blur_ratio)
    if kernel_size % 2 == 0: kernel_size += 1
    blurred_bg = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    flattened = cv2.divide(gray, blurred_bg, scale=255)
    _, thresh = cv2.threshold(flattened, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    c = max(contours, key=cv2.contourArea)
    if _score_candidate(c, params, image.shape) > params.min_score_threshold:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, params.approx_poly_epsilon_ratio * peri, True)
        return approx.reshape(4, 2).astype(np.float32)
    return None


def stage4_feature_detector(image: np.ndarray, params: Params) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=params.s4_max_corners, qualityLevel=params.s4_quality_level,
                                      minDistance=params.s4_min_distance)
    if corners is None or len(corners) < 4: return None
    hull = cv2.convexHull(corners)
    if _score_candidate(hull, params, image.shape) > params.min_score_threshold:
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, params.approx_poly_epsilon_ratio * peri, True)
        if len(approx) == 4:
            return _order_points(approx.reshape(4, 2)).astype(np.float32)
    return None


def stage5_content_analyzer(image: np.ndarray, params: Params) -> Optional[np.ndarray]:
    h, w = image.shape[:2];
    total_area = h * w
    scale = 300 / max(h, w)  # s4_resize_longest_edge is now part of Params, let's use it
    small_img = cv2.resize(image, (int(w * scale), int(h * scale)))
    pixels = small_img.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, params.s4_kmeans_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = centers.astype(np.uint8)
    lab_centers = cv2.cvtColor(centers.reshape(1, -1, 3), cv2.COLOR_BGR2LAB)[0]
    brightest_idx = np.argmax([c[0] for c in lab_centers])
    mask = (labels.reshape(small_img.shape[:2]) == brightest_idx).astype(np.uint8) * 255
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    c = max(contours, key=cv2.contourArea)
    if not (params.score_min_area_ratio < cv2.contourArea(c) / total_area < params.score_max_area_ratio): return None
    rect = cv2.minAreaRect(c)
    return cv2.boxPoints(rect).astype(np.float32)


def stage6_line_reconstructor(image: np.ndarray, params: Params) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, params.canny_min, params.canny_max)
    lines = cv2.HoughLines(edges, 1, np.pi / 180,
                           int(min(image.shape[:2]) / params.s5_hough_threshold_ratio))  # s6 olacak
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
# 4. Ana Bileşen
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
        work_img = cv2.resize(src_img_orig, (int(w * scale), int(h * scale)))

        # Akıllı Ön İşleme
        gray_work_img = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)
        if np.mean(gray_work_img) < 85:  # Karanlık ise
            print("Karanlık görüntü tespit edildi, ekstra kontrast artırılıyor...")
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray_work_img = clahe.apply(gray_work_img)
            work_img = cv2.cvtColor(gray_work_img, cv2.COLOR_GRAY2BGR)  # Tekrar renkliye çevir

        work_img = _unsharp_mask(work_img, self.params.unsharp_strength)

        document_quad = None
        warped = None

        # Parametreleri düzelt: s5_hough_threshold_ratio -> s6_...
        # stage3_boundary_watcher s3_blur_ratio olmalı
        # stage4_feature_detector s4_... olmalı
        # stage5_content_analyzer s5_... olmalı
        # Bu hataları düzeltmek için Params sınıfını ve fonksiyon imzalarını yeniden düzenleyeceğim.

        strategies = {
            "Hızlı Gözcü": stage1_fast_and_simple,
            "Düşük Kontrast Uzmanı": stage2_low_contrast_specialist,
            "Sınır Gözcüsü": stage3_boundary_watcher,
            "Noktasal Köşe Avcısı": stage4_feature_detector,
            "İçerik Analisti": stage5_content_analyzer,
            "Çizgi Dedektifi": stage6_line_reconstructor,
        }

        for name, strategy in strategies.items():
            print(f"Aşama ( {name} ) deneniyor...")
            candidate_quad_scaled = strategy(work_img, self.params)
            if candidate_quad_scaled is not None:
                document_quad = candidate_quad_scaled / scale
                warped_candidate = _four_point_transform(src_img_orig, document_quad)
                if warped_candidate is not None:
                    print(f"Başarılı: Belge '{name}' stratejisi ile bulundu.")
                    warped = warped_candidate
                    break

        if warped is None:
            print("Tüm uzmanlar başarısız. Fallback olarak tüm görüntü kullanılıyor.")
            warped = src_img_orig
            document_quad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)
        self.context["src_quad"] = document_quad.tolist()
        self.context["output_size"] = [warped.shape[1], warped.shape[0]]
        return build_response(context=self)


if __name__ == "__main__":
    Executor(sys.argv[1]).run()