import os
import sys
import cv2
import numpy as np
import math
from typing import Optional, List, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveCorrection.src.utils.response import build_response
from components.PerspectiveCorrection.src.models.PackageModel import PackageModel


class PerspectiveCorrection(Component):
    """
    İki aşamalı hibrit yaklaşım kullanan bileşen:
    1. Hızlı kontur tespiti.
    2. Başarısız olursa, güçlü kenar kesişimi (Hough) tespiti.
    """

    class Params:
        """Algoritma için birleştirilmiş parametreler."""

        def __init__(self, config=None):
            config = config or {}
            # Genel
            self.resize_height = config.get("resize_height", 500)
            self.min_area_ratio = config.get("min_area_ratio", 0.05)  # Esnek alan oranı
            # Kontur Stratejisi
            self.approx_poly_epsilon_ratio = config.get("approx_poly_epsilon_ratio", 0.02)
            # Hough Stratejisi
            self.hough_line_threshold = config.get("hough_line_threshold", 50)
            self.hough_min_line_length = config.get("hough_min_line_length", 50)
            self.hough_max_line_gap = config.get("hough_max_line_gap", 10)
            self.hough_angle_tolerance = config.get("hough_angle_tolerance", 10)

    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.context = {}
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")
        params_data = self.request.get_param("params", {})
        self.params = self.Params(params_data)

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    # --- GEOMETRİ VE YARDIMCI METOTLAR ---
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        pts = pts.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1);
        rect[0] = pts[np.argmin(s)];
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1);
        rect[1] = pts[np.argmin(diff)];
        rect[3] = pts[np.argmax(diff)]
        return rect

    def _four_point_transform(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        rect = self._order_points(pts)
        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl);
        widthB = np.linalg.norm(tr - tl)
        heightA = np.linalg.norm(tr - br);
        heightB = np.linalg.norm(tl - bl)
        maxWidth = int(max(widthA, widthB));
        maxHeight = int(max(heightA, heightB))
        dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
        M = cv.getPerspectiveTransform(rect, dst)
        return cv.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv.INTER_LANCZOS4)

    def _line_intersection(self, line1, line2) -> Optional[Tuple[int, int]]:
        x1, y1, x2, y2 = line1[0];
        x3, y3, x4, y4 = line2[0]
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0: return None
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        ix = int(x1 + t * (x2 - x1));
        iy = int(y1 + t * (y2 - y1))
        return (ix, iy)

    # --- TESPİT STRATEJİLERİ ---
    def _find_quad_with_contours(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Klasik kontur bulma yöntemini uygular."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, self.params.approx_poly_epsilon_ratio * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2).astype(np.float32)
        return None

    def _find_quad_with_hough(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Kenar kesişimi (Hough) yöntemini uygular."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edged, 1, np.pi / 180,
                                threshold=self.params.hough_line_threshold,
                                minLineLength=self.params.hough_min_line_length,
                                maxLineGap=self.params.hough_max_line_gap)
        if lines is None: return None

        horizontal, vertical = [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
            if angle < self.params.hough_angle_tolerance or abs(angle - 180) < self.params.hough_angle_tolerance:
                horizontal.append(line)
            elif abs(angle - 90) < self.params.hough_angle_tolerance:
                vertical.append(line)

        if len(horizontal) < 2 or len(vertical) < 2: return None

        horizontal.sort(key=lambda line: line[0][1]);
        vertical.sort(key=lambda line: line[0][0])
        top_line, bottom_line = horizontal[0], horizontal[-1]
        left_line, right_line = vertical[0], vertical[-1]

        tl = self._line_intersection(top_line, left_line)
        tr = self._line_intersection(top_line, right_line)
        bl = self._line_intersection(bottom_line, left_line)
        br = self._line_intersection(bottom_line, right_line)

        if all((tl, tr, bl, br)):
            return np.array([tl, tr, br, bl], dtype=np.float32)
        return None

    # --- ANA İŞ AKIŞI ---
    def run(self):
        # 1. Hazırlık
        img_obj = Image.get_frame(img=self.image, redis_db=self.redis_db)
        if img_obj is None or img_obj.value is None: raise ValueError("Girdi görüntüsü alınamadı.")

        src_img_orig = img_obj.value
        # Görüntü hazırlama (normalize, renk kanalları vb.)
        if src_img_orig.dtype != np.uint8: src_img_orig = cv2.normalize(src_img_orig, None, 0, 255,
                                                                        cv2.NORM_MINMAX).astype(np.uint8)
        if src_img_orig.ndim == 2:
            src_img_orig = cv2.cvtColor(src_img_orig, cv2.COLOR_GRAY2BGR)
        elif src_img_orig.shape[-1] == 4:
            src_img_orig = cv2.cvtColor(src_img_orig, cv2.COLOR_BGRA2BGR)

        h_orig, w_orig = src_img_orig.shape[:2]
        ratio = h_orig / self.params.resize_height
        work_img = cv2.resize(src_img_orig, (int(w_orig / ratio), self.params.resize_height))

        # 2. AŞAMA 1: Hızlı Kontur Yöntemini Dene
        print("Aşama 1: Hızlı kontur tespiti deneniyor...")
        document_quad_scaled = self._find_quad_with_contours(work_img)

        # 3. AŞAMA 2: Eğer Gerekirse Kenar Kesişimi Yöntemini Dene
        if document_quad_scaled is None:
            print("Kontur yöntemi başarısız. Aşama 2: Kenar kesişimi (Hough) yöntemi deneniyor...")
            document_quad_scaled = self._find_quad_with_hough(work_img)

        warped = None
        document_quad_orig = None

        # 4. Perspektif Düzeltme
        if document_quad_scaled is not None:
            document_quad_orig = document_quad_scaled * ratio

            if cv2.contourArea(document_quad_orig) / (w_orig * h_orig) > self.params.min_area_ratio:
                print("Geçerli bir dörtgen bulundu. Perspektif dönüşümü uygulanıyor...")
                warped = self._four_point_transform(src_img_orig, document_quad_orig)
            else:
                print("Bulunan dörtgen minimum alan oranının altında kaldı.")

        # 5. Sonuçlandırma
        if warped is None:
            print("Geçerli bir belge bulunamadı. Orijinal görüntü kullanılıyor (fallback).")
            warped = src_img_orig
            document_quad_orig = np.array([[0, 0], [w_orig - 1, 0], [w_orig - 1, h_orig - 1], [0, h_orig - 1]],
                                          dtype=np.float32)

        # 6. Çıktıları Kaydetme
        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)
        self.context["src_quad"] = document_quad_orig.tolist()
        self.context["output_size"] = [warped.shape[1], warped.shape[0]]

        print("İşlem tamamlandı.")
        return build_response(context=self)


if __name__ == "__main__":
    Executor(sys.argv[1]).run()