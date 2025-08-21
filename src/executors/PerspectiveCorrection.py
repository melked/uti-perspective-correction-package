import os
import sys
import cv2
import numpy as np
from typing import Optional, List, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveCorrection.src.utils.response import build_response
from components.PerspectiveCorrection.src.models.PackageModel import PackageModel


class PerspectiveCorrection(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.context = {}
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")

        # --- Parametreler ---
        self.RESIZE_HEIGHT = 800
        self.MIN_AREA_RATIO = 0.07
        self.MIN_CONFIDENCE_THRESHOLD = 0.40

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    # --- GEOMETRİ VE YARDIMCI FONKSİYONLAR ---
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

    def _full_image_quad(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

    def _find_quads_from_contours(self, binary_img: np.ndarray, ref_image: np.ndarray) -> List[np.ndarray]:
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return []

        img_area = ref_image.shape[0] * ref_image.shape[1]
        min_area = img_area * self.MIN_AREA_RATIO
        quads = []

        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            if cv2.contourArea(c) < min_area: continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                quads.append(approx.reshape(4, 2).astype(np.float32))
        return quads

    # --- "UZMAN" ÖN İŞLEME STRATEJİLERİ ---
    def _strategy_adaptive_thresh(self, image_gray: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
        return cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)

    def _strategy_canny(self, image_gray: np.ndarray) -> np.ndarray:
        blurred = cv2.bilateralFilter(image_gray, 9, 75, 75)
        v = np.median(blurred);
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * v));
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(blurred, lower, upper)
        return cv2.morphologyEx(edged, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)

    def _strategy_lab_color(self, image_bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        _, thresh = cv2.threshold(cl, 175, 255, cv2.THRESH_BINARY)
        return cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    def _strategy_dark_object(self, image_bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l, _, _ = cv2.split(lab)
        _, thresh = cv2.threshold(l, 80, 255, cv2.THRESH_BINARY_INV)
        return cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    # --- AKILLI PUANLAMA MEKANİZMASI ---
    def _score_quad(self, quad: np.ndarray, image_gray: np.ndarray, edged_image: np.ndarray) -> float:
        h, w = image_gray.shape[:2];
        total_area = w * h
        contour = quad.astype(np.int32)
        area = cv2.contourArea(contour)
        if not (self.MIN_AREA_RATIO < area / total_area < 0.98): return 0.0

        M = cv2.moments(contour);
        if M["m00"] == 0: return 0.0
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        centrality = 1.0 - (np.linalg.norm(np.array([cx, cy]) - np.array([w / 2, h / 2])) / (max(w, h) / 2))

        (tl, tr, br, bl) = self._order_points(quad)
        width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
        height = (np.linalg.norm(tl - bl) + np.linalg.norm(tr - br)) / 2
        if min(width, height) < 1: return 0.0
        aspect_ratio = max(width, height) / min(width, height)
        aspect_score = 1.0 if 1.1 < aspect_ratio < 2.2 else 0.5

        inner_mask = np.zeros(image_gray.shape, dtype="uint8")
        cv2.fillPoly(inner_mask, [contour], 255)
        masked_edges = cv2.bitwise_and(edged_image, edged_image, mask=inner_mask)
        edge_pixel_count = np.count_nonzero(masked_edges)
        content_score = min((edge_pixel_count / area) / 0.05, 1.0) if area > 0 else 0

        border_mask = cv2.dilate(inner_mask, np.ones((15, 15), np.uint8))
        border_mask = cv2.subtract(border_mask, inner_mask)
        border_pixels = image_gray[border_mask == 255]
        border_score = 0
        if border_pixels.size > 100:
            border_std_dev = np.std(border_pixels)
            border_score = 1.0 if 10 < border_std_dev < 50 else 0.2

        final_score = (content_score * 0.4) + (border_score * 0.3) + \
                      (centrality * 0.15) + (aspect_score * 0.1) + (area / total_area * 0.05)
        return final_score

    # --- ANA İŞ AKIŞI ---
    def run(self):
        img_obj = Image.get_frame(img=self.image, redis_db=self.redis_db)
        if img_obj is None or img_obj.value is None: raise ValueError("Girdi görüntüsü alınamadı.")

        src_img_orig = img_obj.value
        if src_img_orig.dtype != np.uint8: src_img_orig = cv2.normalize(src_img_orig, None, 0, 255,
                                                                        cv2.NORM_MINMAX).astype(np.uint8)
        if src_img_orig.ndim == 2:
            src_img_orig = cv2.cvtColor(src_img_orig, cv2.COLOR_GRAY2BGR)
        elif src_img_orig.shape[-1] == 4:
            src_img_orig = cv2.cvtColor(src_img_orig, cv2.COLOR_BGRA2BGR)

        h, w = src_img_orig.shape[:2]
        scale = self.RESIZE_HEIGHT / h if h > self.RESIZE_HEIGHT else 1
        work_img = cv2.resize(src_img_orig, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        work_img_gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)

        # Puanlama için genel kenar haritasını başta oluştur
        work_img_edged = cv2.Canny(work_img_gray, 30, 100)

        # 1. Adım: Tüm Uzman Stratejileri Çalıştır ve Adayları Topla
        print("Uzmanlar komitesi adayları topluyor...")
        all_candidates = []

        # Her strateji, kendi ideal binary görüntüsünü üretir
        binary_canny = self._strategy_canny(work_img_gray)
        binary_adaptive = self._strategy_adaptive_thresh(work_img_gray)
        binary_lab = self._strategy_lab_color(work_img)
        binary_dark = self._strategy_dark_object(work_img)

        # Her binary görüntüden dörtgen adaylarını çıkar
        all_candidates.extend(self._find_quads_from_contours(binary_canny, work_img))
        all_candidates.extend(self._find_quads_from_contours(binary_adaptive, work_img))
        all_candidates.extend(self._find_quads_from_contours(binary_lab, work_img))
        all_candidates.extend(self._find_quads_from_contours(binary_dark, work_img))

        # Tekrarlanan adayları kaldır
        unique_candidates = []
        if all_candidates:
            unique_candidates.append(all_candidates[0])
            for cand in all_candidates[1:]:
                if not any(np.allclose(cand, uc, atol=15) for uc in unique_candidates):
                    unique_candidates.append(cand)

        document_quad = None
        if not unique_candidates:
            print("Hiçbir uzman aday bulamadı.")
            warped = None
        else:
            # 2. Adım: Akıllı Puanlama ile En İyi Adayı Seç
            print(f"{len(unique_candidates)} benzersiz aday bulundu. Komite başkanı değerlendiriyor...")
            scored_candidates = [
                (self._score_quad(q, work_img_gray, work_img_edged), q) for q in unique_candidates
            ]
            scored_candidates.sort(key=lambda x: x[0], reverse=True)

            best_score, best_quad = scored_candidates[0]
            if best_score > self.MIN_CONFIDENCE_THRESHOLD:
                print(f"En iyi aday {best_score:.2f} puanla seçildi.")
                document_quad = best_quad
            else:
                print(
                    f"En iyi adayın puanı ({best_score:.2f}) minimum eşiğin ({self.MIN_CONFIDENCE_THRESHOLD}) altında kaldı.")

        # 3. Adım: Sonuçları İşle
        if document_quad is not None:
            document_quad /= scale
            warped = self._four_point_transform(src_img_orig, document_quad)
        else:
            print("Geçerli bir belge bulunamadı. Fallback olarak orijinal görüntü kullanılıyor.")
            warped = src_img_orig
            document_quad = self._full_image_quad(src_img_orig)

        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)
        self.context["src_quad"] = document_quad.tolist()
        self.context["output_size"] = [warped.shape[1], warped.shape[0]]
        return build_response(context=self)


if __name__ == "__main__":
    Executor(sys.argv[1]).run()