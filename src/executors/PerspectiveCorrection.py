import os
import sys
import cv2
import numpy as np
import math
from collections import defaultdict
from typing import Optional, List, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveCorrection.src.utils.response import build_response
from components.PerspectiveCorrection.src.models.PackageModel import PackageModel


# -----------------------------------------------------------------------------
# 1. Parametre Yönetimi Sınıfı
# -----------------------------------------------------------------------------
class Params:
    """ Tüm uzmanların ve yardımcı fonksiyonların kullandığı parametreleri merkezi olarak yönetir. """

    def __init__(self, config=None):
        config = config or {}
        # Genel ve Stage 1-2 için
        self.canny_min = config.get("canny_min", 50)
        self.canny_max = config.get("canny_max", 150)
        self.contour_min_area_ratio = config.get("contour_min_area_ratio", 0.05)
        self.contour_max_area_ratio = config.get("contour_max_area_ratio", 0.95)
        self.approx_poly_epsilon_ratio = config.get("approx_poly_epsilon_ratio", 0.02)
        # Stage 2 için
        self.boundary_blur_ratio = config.get("boundary_blur_ratio", 5)
        # YENİ UZMAN (Stage 3) için
        self.feature_max_corners = config.get("feature_max_corners", 40)
        self.feature_quality_level = config.get("feature_quality_level", 0.01)
        self.feature_min_distance = config.get("feature_min_distance", 20)
        # Stage 4 için
        self.kmeans_clusters = config.get("kmeans_clusters", 3)
        # Stage 5 için
        self.hough_threshold_ratio = config.get("hough_threshold_ratio", 4)


# -----------------------------------------------------------------------------
# 2. Geometri ve Puanlama Yardımcıları
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


# ... (Diğer yardımcı fonksiyonlar _line_intersection vb. aynı kalır)

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
    area = cv2.contourArea(c);
    total_area = image.shape[0] * image.shape[1]
    if not (params.contour_min_area_ratio < area / total_area < params.contour_max_area_ratio): return None
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, params.approx_poly_epsilon_ratio * peri, True)
    if len(approx) == 4 and cv2.isContourConvex(approx):
        return approx.reshape(4, 2).astype(np.float32)
    return None


def stage2_boundary_watcher(image: np.ndarray, params: Params) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel_size = int(min(image.shape[:2]) / params.boundary_blur_ratio)
    if kernel_size % 2 == 0: kernel_size += 1
    blurred_bg = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    flattened = cv2.divide(gray, blurred_bg, scale=255)
    _, thresh = cv2.threshold(flattened, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    # ... (stage1'deki gibi kontur bulma ve seçme mantığı buraya da uygulanabilir)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    c = max(contours, key=cv2.contourArea)  # Bu aşamada en büyük kontura güvenmek daha mantıklı
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, params.approx_poly_epsilon_ratio * peri, True)
    if len(approx) == 4 and cv2.isContourConvex(approx):
        return approx.reshape(4, 2).astype(np.float32)
    return None


# YENİ UZMAN
def stage3_feature_detector(image: np.ndarray, params: Params) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Doğrudan en güçlü köşe noktalarını bul
    corners = cv2.goodFeaturesToTrack(blurred,
                                      maxCorners=params.feature_max_corners,
                                      qualityLevel=params.feature_quality_level,
                                      minDistance=params.feature_min_distance)
    if corners is None or len(corners) < 4:
        return None

    # En dıştaki 4 köşeyi seç
    corners = np.squeeze(corners)
    hull = cv2.convexHull(corners)

    # Convex Hull'u bir dörtgene yaklaştır
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, params.approx_poly_epsilon_ratio * peri, True)

    if len(approx) == 4:
        return _order_points(approx.reshape(4, 2)).astype(np.float32)
    return None


# ... (stage4_content_analyzer ve stage5_line_reconstructor da eklenebilir)

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
        # ... (kod aynı)
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
            # "İçerik Analisti": stage4_content_analyzer,
            # "Çizgi Dedektifi": stage5_line_reconstructor,
        }

        for name, strategy in strategies.items():
            print(f"Aşama ( {name} ) deneniyor...")
            candidate_quad = strategy(src_img, self.params)

            if candidate_quad is not None:
                warped_candidate = _four_point_transform(src_img, candidate_quad)
                if warped_candidate is not None:
                    print(f"Başarılı: Belge '{name}' stratejisi ile bulundu.")
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
# 5. Çalıştırıcı
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    Executor(sys.argv[1]).run()