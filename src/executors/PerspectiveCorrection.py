import os
import sys
import cv2
import numpy as np
import math
from typing import Optional, List, Tuple, Dict

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveCorrection.src.utils.response import build_response
from components.PerspectiveCorrection.src.models.PackageModel import PackageModel


class PerspectiveCorrection(Component):
    class Params:
        """Tüm algoritma parametrelerini yönetmek için iç içe sınıf."""

        def __init__(self, config=None):
            config = config or {}
            self.resize_longest_edge = config.get("resize_longest_edge", 1200)
            self.unsharp_strength = config.get("unsharp_strength", 1.5)
            self.score_min_area_ratio = config.get("score_min_area_ratio", 0.10)
            self.score_max_area_ratio = config.get("score_max_area_ratio", 0.98)
            self.approx_poly_epsilon_ratio = config.get("approx_poly_epsilon_ratio", 0.02)
            self.min_confidence_threshold = config.get("min_confidence_threshold", 0.40)
            self.hough_line_threshold = config.get("hough_line_threshold", 50)
            self.hough_min_line_length = config.get("hough_min_line_length", 50)
            self.hough_max_line_gap = config.get("hough_max_line_gap", 20)
            self.hough_angle_tolerance = config.get("hough_angle_tolerance", 10)
            self.glare_threshold = config.get("glare_threshold", 240)
            self.consensus_distance_threshold = config.get("consensus_distance_threshold", 0.08)

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

    # --- YARDIMCI VE GEOMETRİ METOTLARI ---
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

    def _four_point_transform(self, image: np.ndarray, pts: np.ndarray) -> Optional[np.ndarray]:
        rect = self._order_points(pts)
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

    def _line_intersection(self, line1, line2) -> Optional[Tuple[int, int]]:
        x1, y1, x2, y2 = line1[0];
        x3, y3, x4, y4 = line2[0]
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0: return None
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        ix = int(x1 + t * (x2 - x1));
        iy = int(y1 + t * (y2 - y1))
        return (ix, iy)

    # --- ÖN İŞLEME VE TEŞHİS ---
    def _diagnose_image(self, image: np.ndarray) -> Dict:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean, std_dev = np.mean(gray), np.std(gray)
        bright_pixels = np.sum(gray > self.params.glare_threshold) / gray.size
        diagnostics = {
            "contrast": "low" if std_dev < 40 else "normal",
            "has_glare": bright_pixels > 0.05,
        }
        print(f"Görüntü Teşhisi: {diagnostics}")
        return diagnostics

    def _preprocess_image(self, image: np.ndarray, diagnostics: Dict) -> np.ndarray:
        processed = cv2.bilateralFilter(image, 9, 75, 75)
        if diagnostics["has_glare"]:
            print("Parlama tespit edildi, düzeltiliyor...")
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            _, glare_mask = cv2.threshold(gray, self.params.glare_threshold, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            glare_mask = cv2.morphologyEx(glare_mask, cv2.MORPH_DILATE, kernel)
            processed = cv2.inpaint(processed, glare_mask, 3, cv2.INPAINT_TELEA)
        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return cv2.addWeighted(processed, 1.0 + self.params.unsharp_strength, cv2.GaussianBlur(processed, (0, 0), 3),
                               -self.params.unsharp_strength, 0)

    # --- ADAY ÜRETME STRATEJİLERİ ---
    def _strategy_canny(self, image: np.ndarray) -> List[np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 50, 150)
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
                                  iterations=3)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return self._get_quads_from_contours(contours)

    def _strategy_hough(self, image: np.ndarray) -> List[np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edged, 1, np.pi / 180, threshold=self.params.hough_line_threshold,
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

    def _get_quads_from_contours(self, contours: List[np.ndarray]) -> List[np.ndarray]:
        quads = []
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, self.params.approx_poly_epsilon_ratio * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                quads.append(approx.reshape(4, 2).astype(np.float32))
        return quads

    # --- ADAY DEĞERLENDİRME VE SEÇME ---
    def _score_quad(self, quad: np.ndarray, image_gray: np.ndarray, edged_image: np.ndarray) -> float:
        h, w = image_gray.shape;
        total_area = h * w
        contour = quad.astype(np.int32)
        area = cv2.contourArea(contour)
        if not (self.params.score_min_area_ratio < area / total_area < self.params.score_max_area_ratio): return 0.0

        # Geometri ve İçerik Puanları... (Önceki kod ile aynı mantık)
        M = cv2.moments(contour);
        cx, cy = (M["m10"] / M["m00"], M["m01"] / M["m00"]) if M["m00"] != 0 else (0, 0)
        centrality = 1.0 - (np.linalg.norm(np.array([cx, cy]) - np.array([w / 2, h / 2])) / (max(w, h) / 2))
        width = (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2
        height = (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) / 2
        aspect_ratio = max(width, height) / (min(width, height) + 1e-6)
        aspect_score = 1.0 if 1.1 < aspect_ratio < 2.2 else 0.5

        mask = np.zeros(image_gray.shape, dtype="uint8");
        cv2.fillPoly(mask, [contour], 255)
        edge_pixel_count = np.count_nonzero(cv2.bitwise_and(edged_image, edged_image, mask=mask))
        content_score = min((edge_pixel_count / area) / 0.05, 1.0) if area > 0 else 0

        return (content_score * 0.6) + (centrality * 0.2) + (aspect_score * 0.2)

    def _select_best_candidate(self, candidates: List[np.ndarray], image_gray: np.ndarray) -> Optional[np.ndarray]:
        if not candidates: return None
        edged_for_scoring = cv2.Canny(image_gray, 50, 150)
        scored_candidates = [(self._score_quad(q, self.params, image_gray, edged_for_scoring), q) for q in candidates]
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_quad = scored_candidates[0]
        print(f"En iyi adayın puanı: {best_score:.2f}")
        if best_score > self.params.min_confidence_threshold: return best_quad
        return None

    # --- ANA İŞ AKIŞI: ORKESTRA ŞEFİ ---
    def run(self):
        # 1. Hazırlık
        img_obj = Image.get_frame(img=self.image, redis_db=self.redis_db)
        if img_obj is None or img_obj.value is None: raise ValueError("Girdi görüntüsü alınamadı.")
        src_img_orig = img_obj.value
        h, w = src_img_orig.shape[:2]
        scale = self.params.resize_longest_edge / max(h, w) if max(h, w) > self.params.resize_longest_edge else 1
        work_img = cv2.resize(src_img_orig, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        # 2. Teşhis ve Ön İşleme
        diagnostics = self._diagnose_image(work_img)
        processed_img = self._preprocess_image(work_img, diagnostics)

        # 3. Aday Üretme
        print("Adaylar üretiliyor...")
        candidates = []
        candidates.extend(self._strategy_canny(processed_img))
        candidates.extend(self._strategy_hough(processed_img))

        # Tekrarlanan adayları kaldır
        unique_candidates = []
        if candidates:
            unique_candidates.append(candidates[0])
            for cand in candidates[1:]:
                if not any(np.allclose(cand, uc, atol=20) for uc in unique_candidates):
                    unique_candidates.append(cand)

        # 4. En İyi Adayı Seçme
        print(f"{len(unique_candidates)} benzersiz aday değerlendiriliyor...")
        best_quad = self._select_best_candidate(unique_candidates, cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY))

        # 5. Sonuçlandırma
        warped = None
        if best_quad is not None:
            best_quad /= scale
            warped = self._four_point_transform(src_img_orig, best_quad)

        if warped is None:
            print("Geçerli bir belge bulunamadı. Orijinal görüntü kullanılıyor.")
            warped = src_img_orig
            best_quad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

        # 6. Kaydetme
        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)
        self.context["src_quad"] = best_quad.tolist()
        self.context["output_size"] = [warped.shape[1], warped.shape[0]]
        return build_response(context=self)


if __name__ == "__main__":
    Executor(sys.argv[1]).run()