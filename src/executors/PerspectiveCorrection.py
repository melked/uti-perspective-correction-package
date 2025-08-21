import os
import sys
import cv2
import numpy as np
import math
from typing import Optional, List, Tuple

# Sisteminize uygun import yolları
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))
from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveCorrection.src.utils.response import build_response
from components.PerspectiveCorrection.src.models.PackageModel import PackageModel


class PerspectiveCorrection(Component):
    """
    Klasik ve gelişmiş tespit yöntemlerini birleştiren hibrit (hibrit) bir yaklaşımla,
    her türlü ortamda belge tespiti ve perspektif düzeltmesi yapan nihai bileşen.
    """

    class Params:
        """Tüm algoritma parametrelerini yönetmek için iç içe sınıf."""

        def __init__(self, config=None):
            config = config or {}
            self.resize_height = config.get("resize_height", 800)
            self.min_area_ratio = config.get("min_area_ratio", 0.1)
            self.approx_poly_epsilon_ratio = config.get("approx_poly_epsilon_ratio", 0.02)
            self.min_confidence_threshold = config.get("min_confidence_threshold", 0.5)
            # Hough Stratejisi için Parametreler
            self.hough_line_threshold = config.get("hough_line_threshold", 50)
            self.hough_min_line_length = config.get("hough_min_line_length", 50)
            self.hough_max_line_gap = config.get("hough_max_line_gap", 15)
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

    # --- 1. Geometri ve Yardımcı Metotlar ---
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
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LANCZOS4)

    def _line_intersection(self, line1, line2) -> Optional[Tuple[int, int]]:
        x1, y1, x2, y2 = line1[0];
        x3, y3, x4, y4 = line2[0]
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0: return None
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        ix = int(x1 + t * (x2 - x1));
        iy = int(y1 + t * (y2 - y1))
        return ix, iy

    # --- 2. "Uzman" Aday Üretme Stratejileri ---
    def _strategy_classic_contours(self, image_gray: np.ndarray) -> List[np.ndarray]:
        """Dokümanlardaki klasik Canny + Kontur yöntemini uygular."""
        print("   -> Uzman 1 (Klasik Kontur) çalışıyor...")
        blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)
        # Kenarlardaki boşlukları birleştirmek için morfolojik kapatma
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return self._get_quads_from_contours(contours)

    def _strategy_hough_intersections(self, image_gray: np.ndarray) -> List[np.ndarray]:
        """Hough çizgilerini kesiştirir (düşük kontrast ve karmaşık zeminler için)."""
        print("   -> Uzman 2 (Hough Kesişim) çalışıyor...")
        edged = cv2.Canny(image_gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edged, 1, np.pi / 180, self.params.hough_line_threshold,
                                minLineLength=self.params.hough_min_line_length,
                                maxLineGap=self.params.hough_max_line_gap)
        if lines is None: return []
        horizontal, vertical = [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
            if angle < self.params.hough_angle_tolerance or abs(angle - 180) < self.params.hough_angle_tolerance:
                horizontal.append(line)
            elif abs(angle - 90) < self.params.hough_angle_tolerance:
                vertical.append(line)
        if len(horizontal) < 2 or len(vertical) < 2: return []
        horizontal.sort(key=lambda l: l[0][1]);
        vertical.sort(key=lambda l: l[0][0])
        tl = self._line_intersection(horizontal[0], vertical[0]);
        tr = self._line_intersection(horizontal[0], vertical[-1])
        bl = self._line_intersection(horizontal[-1], vertical[0]);
        br = self._line_intersection(horizontal[-1], vertical[-1])
        if all((tl, tr, bl, br)): return [np.array([tl, tr, br, bl], dtype=np.float32)]
        return []

    # --- YENİ EKLENEN UZMAN STRATEJİ ---
    def _strategy_adaptive_thresh(self, image_gray: np.ndarray) -> List[np.ndarray]:
        """Düşük kontrast ve değişken ışık uzmanı."""
        print("   -> Uzman 3 (Adaptif Eşikleme) çalışıyor...")
        # Gürültüyü azalt ama kenarları koru
        blurred = cv2.bilateralFilter(image_gray, 11, 17, 17)
        # Yerel koşullara göre eşikleme yap
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
        # Gürültüyü temizle ve hatları birleştir
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return self._get_quads_from_contours(contours)
    # ------------------------------------

    def _get_quads_from_contours(self, contours: List[np.ndarray]) -> List[np.ndarray]:
        """Verilen kontur listesinden 4 köşeli olanları ayıklar."""
        quads = []
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:  # En büyük 5 adaya bakmak yeterli
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, self.params.approx_poly_epsilon_ratio * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                quads.append(approx.reshape(4, 2).astype(np.float32))
        return quads

    # --- 3. Akıllı Puanlama ---
    def _score_candidate(self, quad: np.ndarray, image_gray: np.ndarray) -> float:
        """Bir adayı geometri ve içerik özelliklerine göre puanlar."""
        h, w = image_gray.shape;
        total_area = h * w
        contour = quad.astype(np.int32)
        area = cv2.contourArea(contour)
        if not (self.params.min_area_ratio < area / total_area < 0.98): return 0.0

        M = cv2.moments(contour)
        if M["m00"] == 0: return 0.0
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        centrality = 1.0 - (np.linalg.norm(np.array([cx, cy]) - np.array([w / 2, h / 2])) / (max(w, h) / 2))

        width = (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2
        height = (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) / 2
        aspect_ratio = max(width, height) / (min(width, height) + 1e-6)
        aspect_score = 1.0 if 1.2 < aspect_ratio < 2.0 else 0.5

        mask = np.zeros(image_gray.shape, dtype="uint8");
        cv2.fillPoly(mask, [contour], 255)
        edged_content = cv2.Canny(image_gray, 50, 150)
        edge_pixel_count = np.count_nonzero(cv2.bitwise_and(edged_content, edged_content, mask=mask))
        content_score = min((edge_pixel_count / area) / 0.1, 1.0) if area > 0 else 0

        return (content_score * 0.5) + (centrality * 0.2) + (aspect_score * 0.2) + (area / total_area * 0.1)

    # --- 4. Ana İş Akışı (Orkestra Şefi) ---
    def run(self):
        # Adım 1: Görüntüyü Hazırla
        img_obj = Image.get_frame(img=self.image, redis_db=self.redis_db)
        if img_obj is None or img_obj.value is None: raise ValueError("Girdi görüntüsü alınamadı.")

        src_img_orig = img_obj.value
        if src_img_orig.dtype != np.uint8: src_img_orig = cv2.normalize(src_img_orig, None, 0, 255,
                                                                        cv2.NORM_MINMAX).astype(np.uint8)
        if src_img_orig.ndim == 2:
            src_img_orig = cv2.cvtColor(src_img_orig, cv2.COLOR_GRAY2BGR)
        elif src_img_orig.shape[-1] == 4:
            src_img_orig = cv2.cvtColor(src_img_orig, cv2.COLOR_BGRA2BGR)

        h_orig, w_orig = src_img_orig.shape[:2]
        ratio = h_orig / self.params.resize_height
        work_img = cv2.resize(src_img_orig, (int(w_orig / ratio), self.params.resize_height))
        work_img_gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)

        # Adım 2: Hibrit Uzmanlar Komitesi ile Aday Üret
        print("Hibrit Uzmanlar Komitesi adayları üretiyor...")
        candidates = []
        candidates.extend(self._strategy_classic_contours(work_img_gray))
        candidates.extend(self._strategy_hough_intersections(work_img_gray))
        # YENİ UZMANI BURADA ÇAĞIRIYORUZ
        candidates.extend(self._strategy_adaptive_thresh(work_img_gray))

        document_quad_orig = None
        warped = None

        if not candidates:
            print("Hiçbir uzman aday bulamadı.")
        else:
            # Tekrarlanan adayları kaldır
            unique_candidates = []
            if candidates:
                unique_candidates.append(candidates[0])
                for cand in candidates[1:]:
                    if not any(np.allclose(cand, uc, atol=20) for uc in unique_candidates):
                        unique_candidates.append(cand)

            # Adım 3: Akıllı Puanlama ile En İyi Adayı Seç
            print(f"{len(unique_candidates)} benzersiz aday değerlendiriliyor...")
            scored_candidates = [(self._score_candidate(q, work_img_gray), q) for q in unique_candidates]
            scored_candidates.sort(key=lambda x: x[0], reverse=True)

            best_score, best_quad_scaled = scored_candidates[0]

            if best_score > self.params.min_confidence_threshold:
                print(f"En iyi aday {best_score:.2f} puanla seçildi.")
                document_quad_orig = best_quad_scaled * ratio
                warped = self._four_point_transform(src_img_orig, document_quad_orig)
            else:
                print(f"En iyi adayın puanı ({best_score:.2f}) minimum eşiğin altında kaldı.")

        # Adım 4: Sonuçlandırma (Fallback)
        if warped is None:
            print("Geçerli bir belge bulunamadı. Orijinal görüntü kullanılıyor.")
            warped = src_img_orig
            document_quad_orig = np.array([[0, 0], [w_orig - 1, 0], [w_orig - 1, h_orig - 1], [0, h_orig - 1]],
                                          dtype=np.float32)

        # Adım 5: Çıktıları Sisteme Kaydet
        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)
        self.context["src_quad"] = document_quad_orig.tolist()
        self.context["output_size"] = [warped.shape[1], warped.shape[0]]

        return build_response(context=self)


if __name__ == "__main__":
    Executor(sys.argv[1]).run()