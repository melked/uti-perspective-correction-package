import os
import sys
import cv2
import numpy as np
import math
from typing import Optional, List, Tuple, Dict

# Sisteminize uygun import yolları
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))
from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveCorrection.src.utils.response import build_response
from components.PerspectiveCorrection.src.models.PackageModel import PackageModel


class PerspectiveCorrection(Component):
    class Params:
        def __init__(self, config=None):
            config = config or {}
            # Genel Ayarlar
            self.resize_longest_edge = config.get("resize_longest_edge", 1000)
            self.min_confidence_threshold = config.get("min_confidence_threshold", 0.45)
            self.approx_poly_epsilon_ratio = config.get("approx_poly_epsilon_ratio", 0.02)
            # Ön İşleme Ayarları
            self.unsharp_strength = config.get("unsharp_strength", 1.5)
            self.glare_threshold = config.get("glare_threshold", 235)
            self.shadow_clahe_clip_limit = config.get("shadow_clahe_clip_limit", 2.0)
            # Puanlama Ağırlıkları
            self.w_area = config.get("w_area", 0.20)
            self.w_aspect = config.get("w_aspect", 0.25)
            self.w_angle = config.get("w_angle", 0.30)
            self.w_centrality = config.get("w_centrality", 0.15)
            self.w_parallel = config.get("w_parallel", 0.10)
            # Kenar İnce Ayarı
            self.edge_refinement_radius = config.get("edge_refinement_radius", 10)

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

    # --- 1. Geometri ve Temel Yardımcılar ---
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
        if maxWidth < 20 or maxHeight < 20: return None
        dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LANCZOS4)

    # --- 2. Gelişmiş Teşhis ve Ön İşleme ---
    def _diagnose_image(self, image: np.ndarray) -> Dict:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean, std_dev = np.mean(gray), np.std(gray)
        bright_pixels = np.sum(gray > self.params.glare_threshold) / gray.size
        diagnostics = {
            "contrast": "low" if std_dev < 45 else "normal",
            "has_glare": bright_pixels > 0.02,
        }
        print(f"Görüntü Teşhisi: {diagnostics}")
        return diagnostics

    def _adaptive_preprocess(self, image: np.ndarray, diagnostics: Dict) -> np.ndarray:
        processed = image.copy()
        if diagnostics["has_glare"]:
            print(" -> Parlama tespit edildi, temizleniyor...")
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            _, glare_mask = cv2.threshold(gray, self.params.glare_threshold, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            glare_mask = cv2.dilate(glare_mask, kernel)
            processed = cv2.inpaint(processed, glare_mask, 3, cv2.INPAINT_TELEA)

        print(" -> Gölge ve kontrast telafisi uygulanıyor (LAB-CLAHE)...")
        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.params.shadow_clahe_clip_limit, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        processed = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        blurred = cv2.GaussianBlur(processed, (0, 0), 3)
        return cv2.addWeighted(processed, 1.0 + self.params.unsharp_strength, blurred, -self.params.unsharp_strength, 0)

    # --- 3. Uzman Aday Üretme Stratejileri ---
    def _strategy_canny(self, image: np.ndarray) -> List[np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        edged = cv2.Canny(blurred, 50, 150)
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
                                  iterations=2)
        return self._get_quads_from_contours(closed)

    def _strategy_adaptive(self, image: np.ndarray) -> List[np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 5)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
        return self._get_quads_from_contours(closed)

    def _get_quads_from_contours(self, binary_image: np.ndarray) -> List[np.ndarray]:
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        quads = []
        img_area = binary_image.shape[0] * binary_image.shape[1]
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            if cv2.contourArea(c) / img_area < self.params.score_min_area_ratio: continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, self.params.approx_poly_epsilon_ratio * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                quads.append(approx.reshape(4, 2).astype(np.float32))
        return quads

    # --- 4. Aday İnce Ayarı ve Puanlama ---
    def _refine_quad_edges(self, quad: np.ndarray, image_gray: np.ndarray) -> np.ndarray:
        refined_quad = quad.copy()
        radius = self.params.edge_refinement_radius
        for i, corner in enumerate(quad):
            x, y = int(corner[0]), int(corner[1])
            y_min, y_max = max(0, y - radius), min(image_gray.shape[0], y + radius)
            x_min, x_max = max(0, x - radius), min(image_gray.shape[1], x + radius)
            roi = image_gray[y_min:y_max, x_min:x_max]
            if roi.size == 0: continue
            grad_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = cv2.magnitude(grad_x, grad_y)
            _, _, _, max_loc = cv2.minMaxLoc(magnitude)
            refined_corner = (x_min + max_loc[0], y_min + max_loc[1])
            if np.linalg.norm(np.array(refined_corner) - corner) < radius * 1.5:
                refined_quad[i] = refined_corner
        return refined_quad

    def _score_candidate(self, quad: np.ndarray, image_shape: tuple) -> float:
        h, w = image_shape
        total_area = h * w
        contour = quad.astype(np.int32)
        area = cv2.contourArea(contour)

        # Geometrik Puanlar
        M = cv2.moments(contour);
        cx, cy = (M["m10"] / M["m00"], M["m01"] / M["m00"]) if M["m00"] != 0 else (w / 2, h / 2)
        centrality = 1.0 - (np.linalg.norm(np.array([cx, cy]) - np.array([w / 2, h / 2])) / (max(w, h) / 2))

        (tl, tr, br, bl) = self._order_points(quad)
        top_len = np.linalg.norm(tr - tl);
        bot_len = np.linalg.norm(br - bl)
        left_len = np.linalg.norm(tl - bl);
        right_len = np.linalg.norm(tr - br)
        if min(top_len, bot_len, left_len, right_len) < 1: return 0.0

        parallel_score = 1.0 - (abs(top_len - bot_len) + abs(left_len - right_len)) / (w + h)
        width = (top_len + bot_len) / 2;
        height = (left_len + right_len) / 2
        aspect_ratio = max(width, height) / min(width, height)
        aspect_score = 1.0 if 1.2 < aspect_ratio < 2.0 else 0.5

        # Açı Puanı (90 dereceye yakınlık)
        v1 = tr - tl;
        v2 = br - tr;
        v3 = bl - br;
        v4 = tl - bl
        a1 = math.acos(np.dot(v1, -v4) / (np.linalg.norm(v1) * np.linalg.norm(v4) + 1e-6))
        a2 = math.acos(np.dot(-v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6))
        angle_dev = abs(a1 - math.pi / 2) + abs(a2 - math.pi / 2)
        angle_score = 1.0 - (angle_dev / math.pi)

        # Final Ağırlıklı Puan
        return (self.params.w_area * (area / total_area) +
                self.params.w_aspect * aspect_score +
                self.params.w_angle * angle_score +
                self.params.w_centrality * centrality +
                self.params.w_parallel * parallel_score)

    # --- 5. Ana İş Akışı (Orkestra Şefi) ---
    def run(self):
        # Adım 1: Görüntüyü Hazırla
        img_obj = Image.get_frame(img=self.image, redis_db=self.redis_db)
        if img_obj is None or img_obj.value is None: raise ValueError("Girdi görüntüsü alınamadı.")
        src_img_orig = img_obj.value
        h_orig, w_orig = src_img_orig.shape[:2]
        scale = self.params.resize_longest_edge / max(h_orig, w_orig) if max(h_orig,
                                                                             w_orig) > self.params.resize_longest_edge else 1
        work_img = cv2.resize(src_img_orig, (int(w_orig * scale), int(h_orig * scale)), interpolation=cv2.INTER_AREA)

        # Adım 2: Teşhis ve Adaptif Ön İşleme
        diagnostics = self._diagnose_image(work_img)
        processed_img = self._adaptive_preprocess(work_img, diagnostics)

        # Adım 3: Uzmanlar Komitesi ile Aday Üret
        print("Uzmanlar komitesi adayları üretiyor...")
        candidates = []
        candidates.extend(self._strategy_canny(processed_img))
        candidates.extend(self._strategy_adaptive(processed_img))

        document_quad = None
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

            # Adım 4: Adayları İnce Ayarla ve Puanla
            print(f"{len(unique_candidates)} aday ince ayarlanıyor ve puanlanıyor...")
            work_img_gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)
            refined_candidates = [self._refine_quad_edges(q, work_img_gray) for q in unique_candidates]
            scored_candidates = [(self._score_candidate(q, (work_img.shape[0], work_img.shape[1])), q) for q in
                                 refined_candidates]
            scored_candidates.sort(key=lambda x: x[0], reverse=True)

            best_score, best_quad_scaled = scored_candidates[0]

            if best_score > self.params.min_confidence_threshold:
                print(f"En iyi aday {best_score:.2f} puanla seçildi.")
                document_quad = best_quad_scaled / scale
                warped = self._four_point_transform(src_img_orig, document_quad)
            else:
                print(f"En iyi adayın puanı ({best_score:.2f}) minimum eşiğin altında kaldı.")

        if warped is None:
            print("Geçerli bir belge bulunamadı. Fallback olarak orijinal görüntü kullanılıyor.")
            warped = src_img_orig
            document_quad = np.array([[0, 0], [w_orig - 1, 0], [w_orig - 1, h_orig - 1], [0, h_orig - 1]],
                                     dtype=np.float32)

        # Adım 5: Çıktıları Kaydet
        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)
        self.context["src_quad"] = document_quad.tolist()
        self.context["output_size"] = [warped.shape[1], warped.shape[0]]
        return build_response(context=self)


if __name__ == "__main__":
    Executor(sys.argv[1]).run()
    Executor(sys.argv[1]).run()