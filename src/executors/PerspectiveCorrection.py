import os
import sys
import cv2
import numpy as np
from PIL import Image as PILImage
from dataclasses import dataclass
from typing import Optional, Tuple, List
from abc import ABC, abstractmethod

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveCorrection.src.utils.response import build_response
from components.PerspectiveCorrection.src.models.PackageModel import PackageModel


@dataclass
class DetectionParams:
    """Algılama parametreleri"""
    clahe_clip: float = 3.0
    clahe_grid: Tuple[int, int] = (8, 8)
    gamma_auto: bool = True
    gamma_value: float = 1.0
    canny_low: int = 50
    canny_high: int = 150
    min_area_ratio: float = 0.1  # Görüntünün minimum %10'u
    quality_threshold: float = 0.01
    min_distance_ratio: float = 0.05  # Görüntü boyutunun %5'i


class ImageProcessor:
    """Görüntü ön işleme sınıfı"""

    @staticmethod
    def adaptive_gamma(img: np.ndarray) -> float:
        """Görüntü parlaklığına göre otomatik gamma hesapla"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        brightness = gray.mean() / 255.0

        if brightness < 0.3:
            return 1.8  # Karanlık için gamma yükselt
        elif brightness > 0.7:
            return 0.6  # Parlak için gamma düşür
        return 1.0

    @staticmethod
    def enhance_image(img: np.ndarray, params: DetectionParams) -> np.ndarray:
        """Görüntüyü köşe algılama için optimize et"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        # CLAHE uygula
        clahe = cv2.createCLAHE(clipLimit=params.clahe_clip, tileGridSize=params.clahe_grid)
        enhanced = clahe.apply(gray)

        # Gamma düzeltmesi
        gamma = ImageProcessor.adaptive_gamma(img) if params.gamma_auto else params.gamma_value
        table = np.array([(i / 255.0) ** (1.0 / gamma) * 255 for i in range(256)]).astype(np.uint8)
        gamma_corrected = cv2.LUT(enhanced, table)

        # Hafif bulanıklaştırma
        return cv2.GaussianBlur(gamma_corrected, (5, 5), 0)


class CornerDetector(ABC):
    """Köşe algılama sınıflarının base class'ı"""

    @abstractmethod
    def detect(self, img: np.ndarray, params: DetectionParams) -> Optional[np.ndarray]:
        pass


class GoodFeaturesDetector(CornerDetector):
    """GoodFeaturesToTrack tabanlı algılama"""

    def detect(self, img: np.ndarray, params: DetectionParams) -> Optional[np.ndarray]:
        h, w = img.shape[:2]
        min_distance = int(min(h, w) * params.min_distance_ratio)

        corners = cv2.goodFeaturesToTrack(
            img,
            maxCorners=16,
            qualityLevel=params.quality_threshold,
            minDistance=min_distance
        )

        if corners is None or len(corners) < 4:
            return None

        corners = np.squeeze(corners)
        return self._select_corner_points(corners)

    def _select_corner_points(self, corners: np.ndarray) -> np.ndarray:
        """En uygun 4 köşeyi seç"""
        if len(corners) <= 4:
            return corners

        # En dış 4 köşeyi seç
        center = np.mean(corners, axis=0)
        distances = np.linalg.norm(corners - center, axis=1)
        indices = np.argsort(distances)[-4:]
        return corners[indices]


class ContourDetector(CornerDetector):
    """Contour tabanlı algılama"""

    def detect(self, img: np.ndarray, params: DetectionParams) -> Optional[np.ndarray]:
        h, w = img.shape[:2]
        min_area = h * w * params.min_area_ratio

        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            # Dikdörtgen yaklaşımı
            hull = cv2.convexHull(contour)
            epsilon = 0.015 * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)

            if 4 <= len(approx) <= 6:
                points = approx.reshape(-1, 2)
                if len(points) == 4:
                    return points
                else:
                    # En dış 4 köşeyi seç
                    center = np.mean(points, axis=0)
                    distances = np.linalg.norm(points - center, axis=1)
                    indices = np.argsort(distances)[-4:]
                    return points[indices]

        return None


class HoughLinesDetector(CornerDetector):
    """Hough Lines tabanlı algılama"""

    def detect(self, img: np.ndarray, params: DetectionParams) -> Optional[np.ndarray]:
        h, w = img.shape[:2]

        # Hough Lines ile yatay ve dikey çizgileri bul
        lines = cv2.HoughLines(img, 1, np.pi / 180, min(h, w) // 4)

        if lines is None or len(lines) < 4:
            return None

        horizontal_lines, vertical_lines = self._separate_lines(lines)

        if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
            return None

        # Kesişim noktalarını hesapla
        intersections = self._find_intersections(horizontal_lines[:2], vertical_lines[:2], w, h)

        return intersections if len(intersections) == 4 else None

    def _separate_lines(self, lines: np.ndarray) -> Tuple[List, List]:
        """Çizgileri yatay ve dikey olarak ayır"""
        horizontal, vertical = [], []

        for rho, theta in lines[:20, 0]:  # İlk 20 çizgi
            angle = theta * 180 / np.pi
            if abs(angle) < 15 or abs(angle - 180) < 15:
                horizontal.append((rho, theta))
            elif abs(angle - 90) < 15:
                vertical.append((rho, theta))

        return horizontal, vertical

    def _find_intersections(self, h_lines: List, v_lines: List, w: int, h: int) -> np.ndarray:
        """Çizgi kesişimlerini hesapla"""
        intersections = []

        for rho1, theta1 in h_lines:
            for rho2, theta2 in v_lines:
                cos1, sin1 = np.cos(theta1), np.sin(theta1)
                cos2, sin2 = np.cos(theta2), np.sin(theta2)
                det = cos1 * sin2 - sin1 * cos2

                if abs(det) > 1e-6:
                    x = (sin2 * rho1 - sin1 * rho2) / det
                    y = (cos1 * rho2 - cos2 * rho1) / det

                    if 0 <= x < w and 0 <= y < h:
                        intersections.append([x, y])

        return np.array(intersections) if intersections else np.array([])


class CornerOrderer:
    """Köşe sıralama sınıfı"""

    @staticmethod
    def order_points(points: np.ndarray) -> np.ndarray:
        """Köşeleri saat yönünde sırala: üst-sol, üst-sağ, alt-sağ, alt-sol"""
        if len(points) != 4:
            raise ValueError("Tam olarak 4 köşe gerekli")

        # Toplamlarına göre üst-sol ve alt-sağ köşeleri bul
        point_sums = points.sum(axis=1)
        top_left = points[np.argmin(point_sums)]
        bottom_right = points[np.argmax(point_sums)]

        # Farklarına göre üst-sağ ve alt-sol köşeleri bul
        point_diffs = np.diff(points, axis=1)
        top_right = points[np.argmin(point_diffs)]
        bottom_left = points[np.argmax(point_diffs)]

        return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


class QualityEvaluator:
    """Köşe kalitesi değerlendirme sınıfı"""

    @staticmethod
    def evaluate_corners(corners: np.ndarray, img_shape: Tuple[int, int]) -> float:
        """Köşe kalitesini değerlendir (0-1 arası)"""
        if corners is None or len(corners) != 4:
            return 0.0

        h, w = img_shape[:2]
        ordered_corners = CornerOrderer.order_points(corners)

        # Alan skoru
        area = cv2.contourArea(ordered_corners)
        area_score = min(1.0, area / (w * h))

        # Dikdörtgenlik skoru
        side_lengths = []
        for i in range(4):
            p1, p2 = ordered_corners[i], ordered_corners[(i + 1) % 4]
            side_lengths.append(np.linalg.norm(p2 - p1))

        # Karşılıklı kenarların benzerliği
        ratio1 = min(side_lengths[0], side_lengths[2]) / max(side_lengths[0], side_lengths[2])
        ratio2 = min(side_lengths[1], side_lengths[3]) / max(side_lengths[1], side_lengths[3])
        shape_score = (ratio1 + ratio2) / 2

        return area_score * shape_score


class DocumentDetector:
    """Ana belge algılama sınıfı"""

    def __init__(self, params: Optional[DetectionParams] = None):
        self.params = params or DetectionParams()
        self.detectors = [
            GoodFeaturesDetector(),
            ContourDetector(),
            HoughLinesDetector()
        ]

    def detect_corners(self, img: np.ndarray) -> np.ndarray:
        """En iyi köşe algılama yöntemini kullanarak köşeleri bul"""
        # Görüntüyü ön işleme
        processed = ImageProcessor.enhance_image(img, self.params)
        edges = cv2.Canny(processed, self.params.canny_low, self.params.canny_high)

        best_corners = None
        best_score = 0.0

        # Tüm algılama yöntemlerini dene
        for detector in self.detectors:
            try:
                corners = detector.detect(edges, self.params)
                if corners is not None and len(corners) >= 4:
                    score = QualityEvaluator.evaluate_corners(corners[:4], img.shape)
                    if score > best_score:
                        best_score = score
                        best_corners = corners[:4]
            except Exception:
                continue

        # Hiçbir yöntem başarılı olamazsa fallback
        if best_corners is None:
            best_corners = self._fallback_corners(img.shape)

        return CornerOrderer.order_points(best_corners)

    def _fallback_corners(self, img_shape: Tuple[int, int]) -> np.ndarray:
        """Son çare: görüntü köşelerine yakın noktalar"""
        h, w = img_shape[:2]
        margin = min(h, w) * 0.05
        return np.array([
            [margin, margin],
            [w - margin, margin],
            [w - margin, h - margin],
            [margin, h - margin]
        ])


class PerspectiveTransformer:
    """Perspektif dönüştürme sınıfı"""

    @staticmethod
    def transform(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """4 köşe kullanarak perspektif düzeltme yap"""
        ordered_corners = CornerOrderer.order_points(corners)

        # Hedef boyutları hesapla
        tl, tr, br, bl = ordered_corners

        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        max_width = int(max(width_a, width_b))

        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_height = int(max(height_a, height_b))

        # Hedef köşeler
        dst_corners = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype=np.float32)

        # Perspektif dönüştürme matrisi
        transform_matrix = cv2.getPerspectiveTransform(ordered_corners, dst_corners)

        # Dönüştürme uygula
        warped = cv2.warpPerspective(img, transform_matrix, (max_width, max_height))

        return warped


def correct_perspective(img: np.ndarray, params: Optional[DetectionParams] = None) -> PILImage.Image:
    """Ana perspektif düzeltme fonksiyonu"""
    detector = DocumentDetector(params)
    corners = detector.detect_corners(img)

    if corners is None:
        raise ValueError("Belge köşeleri bulunamadı")

    corrected = PerspectiveTransformer.transform(img, corners)

    # BGR'den RGB'ye çevir (PIL için)
    if len(corrected.shape) == 3:
        corrected = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)

    return PILImage.fromarray(corrected)


class PerspectiveCorrection(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")

        # Parametreleri al
        params_dict = self.request.get_param("params", {})
        self.params = DetectionParams(**params_dict)

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def run(self):
        img = Image.get_frame(img=self.image, redis_db=self.redis_db)
        img_np = img.value

        # Görüntü formatını normalize et
        if img_np.dtype != np.uint8:
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)

        # Perspektif düzeltme
        result_img = correct_perspective(img_np, self.params)

        img.value = np.array(result_img)
        self.image = Image.set_frame(img=img, package_uID=self.uID, redis_db=self.redis_db)

        return build_response(context=self)


if __name__ == "__main__":
    Executor(sys.argv[1]).run()