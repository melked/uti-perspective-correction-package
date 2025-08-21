import os
import sys
import cv2
import numpy as np
from typing import Optional, List, Dict

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveTransformation.src.utils.response import build_response
from components.PerspectiveTransformation.src.models.PackageModel import PackageModel


class PerspectiveTransformation(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.context = {}
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")

        # YENİ: Tüm ayarlar ve parametreler için merkezi bir config yapısı
        # Bu değerler bootstrap config'den de yüklenebilir.
        self.config = {
            "RESIZE_HEIGHT": 500,
            "MIN_AREA_RATIO": 0.15,
            "CANNY_SIGMA": 0.33,
            "MORPH_KERNEL_SIZE": (5, 5),
            "APPROX_POLY_EPSILON_RATIO": 0.02,
            # Puanlama ağırlıkları
            "SCORE_WEIGHT_AREA": 0.4,
            "SCORE_WEIGHT_PERPENDICULARITY": 0.4,
            "SCORE_WEIGHT_ASPECT_RATIO": 0.2,
            "MIN_SCORE_THRESHOLD": 0.2,  # Geçerli bir aday için minimum puan
        }
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.config["MORPH_KERNEL_SIZE"])

    @staticmethod
    def bootstrap(config: dict) -> dict:
        # Gerekirse, component'in başlangıç ayarları buradan yapılabilir.
        return {}

    # --- GÖRÜNTÜ İŞLEME YARDIMCI METOTLARI (Sınıf içine taşındı) ---

    def _prepare_image(self, img: np.ndarray) -> np.ndarray:
        if img is None or img.size == 0:
            raise ValueError("Input image is empty or None.")
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def _resize_image(self, image: np.ndarray, height: int) -> Tuple[np.ndarray, float]:
        """Görüntüyü yeniden boyutlandırır ve orijinal oranı döndürür."""
        h, w = image.shape[:2]
        ratio = h / float(height)
        dim = (int(w / ratio), height)
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        return resized, ratio

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Köşeleri sol-üst, sağ-üst, sağ-alt, sol-alt sırasına göre dizer."""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def _four_point_transform(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Verilen 4 köşe noktasına göre perspektif düzeltme uygular."""
        rect = self._order_points(pts)
        (tl, tr, br, bl) = rect

        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))

        dst = np.array([
            [0, 0], [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LANCZOS4)

    # --- YENİ: BELGE TESPİT MANTIĞI ---

    def _preprocess_for_detection(self, image: np.ndarray) -> np.ndarray:
        """Kenar tespiti için genel ön işleme adımları."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(gray, 9, 75, 75)
        v = np.median(blur)
        sigma = self.config["CANNY_SIGMA"]
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(blur, lower, upper)
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, self.morph_kernel, iterations=2)
        return closed

    def _find_quad_from_contours(self, binary_img: np.ndarray, ref_image: np.ndarray) -> List[np.ndarray]:
        """İkili görüntüden 4 köşeli kontur adaylarını bulur."""
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return []

        img_area = ref_image.shape[0] * ref_image.shape[1]
        min_area = img_area * self.config["MIN_AREA_RATIO"]

        candidates = []
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            if cv2.contourArea(c) < min_area: continue

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, self.config["APPROX_POLY_EPSILON_RATIO"] * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                candidates.append(approx.reshape(4, 2))

        return candidates

    def _detect_candidates(self, image: np.ndarray) -> List[np.ndarray]:
        """Farklı stratejiler kullanarak aday dörtgenleri bulur."""
        candidates = []
        # Strateji 1: Genel Canny Kenar Tespiti
        processed_img = self._preprocess_for_detection(image)
        candidates.extend(self._find_quad_from_contours(processed_img, image))

        # Strateji 2: Adaptif Eşikleme (Farklı ışık koşulları için)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        candidates.extend(self._find_quad_from_contours(thresh, image))

        # Benzersiz adayları tut
        unique_candidates = []
        if candidates:
            unique_candidates.append(candidates[0])
            for cand in candidates[1:]:
                if not any(np.allclose(cand, uc, atol=10) for uc in unique_candidates):
                    unique_candidates.append(cand)
        return unique_candidates

    # --- YENİ VE EN ÖNEMLİ BÖLÜM: ADAY SEÇİMİ ---

    def _score_candidate(self, image: np.ndarray, pts: np.ndarray) -> float:
        """Bir aday dörtgeni geometrik özelliklerine göre puanlar."""
        img_area = image.shape[0] * image.shape[1]
        area_score = cv2.contourArea(pts) / img_area

        ordered_pts = self._order_points(pts)
        (tl, tr, br, bl) = ordered_pts

        v_tr_tl = tr - tl
        v_tr_br = tr - br
        v_bl_tl = bl - tl

        cos_tl = abs(np.dot(v_tr_tl, v_bl_tl) / (np.linalg.norm(v_tr_tl) * np.linalg.norm(v_bl_tl) + 1e-6))
        cos_tr = abs(np.dot(v_tr_tl, v_tr_br) / (np.linalg.norm(v_tr_tl) * np.linalg.norm(v_tr_br) + 1e-6))
        perpendicularity_score = 1 - (cos_tl + cos_tr) / 2.0

        width = np.linalg.norm(tr - tl)
        height = np.linalg.norm(tr - br)
        if width == 0 or height == 0: return 0.0
        aspect_ratio = max(width, height) / min(width, height)
        target_ratio = np.sqrt(2)  # A4
        aspect_ratio_score = max(0, 1 - abs(aspect_ratio - target_ratio) / target_ratio)

        return (area_score * self.config["SCORE_WEIGHT_AREA"] +
                perpendicularity_score * self.config["SCORE_WEIGHT_PERPENDICULARITY"] +
                aspect_ratio_score * self.config["SCORE_WEIGHT_ASPECT_RATIO"])

    def _select_best_candidate(self, image: np.ndarray, candidates: List[np.ndarray]) -> Optional[np.ndarray]:
        """Adaylar arasından en yüksek puanlı olanı seçer."""
        if not candidates:
            return None

        scored_candidates = sorted(
            [(self._score_candidate(image, c), c) for c in candidates],
            key=lambda x: x[0], reverse=True
        )

        best_score, best_candidate = scored_candidates[0]

        print(f"En iyi adayın puanı: {best_score:.4f}")
        if best_score < self.config["MIN_SCORE_THRESHOLD"]:
            return None
        return best_candidate

    # --- ANA ÇALIŞMA METODU (GELİŞTİRİLDİ) ---

    def run(self):
        img_obj = Image.get_frame(img=self.image, redis_db=self.redis_db)
        if img_obj is None or img_obj.value is None:
            raise ValueError("No input image provided or failed to load.")

        src_img = self._prepare_image(img_obj.value)

        # Adım 1: Performans için görüntüyü yeniden boyutlandır
        resized_img, ratio = self._resize_image(src_img, self.config["RESIZE_HEIGHT"])

        # Adım 2: Yeniden boyutlandırılmış görüntü üzerinde adayları tespit et
        candidates = self._detect_candidates(resized_img)

        # Adım 3: En iyi adayı puanlama sistemiyle seç
        best_quad_resized = self._select_best_candidate(resized_img, candidates)

        # Adım 4: Eğer iyi bir aday bulunursa, onu kullan. Bulunmazsa tüm resmi al.
        if best_quad_resized is not None:
            # Köşeleri orijinal görüntü boyutuna geri ölçekle
            best_quad_original = best_quad_resized.astype(np.float32) * ratio
        else:
            print("Uygun bir belge bulunamadı, tüm görüntü işleniyor.")
            h, w = src_img.shape[:2]
            best_quad_original = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

        # Adım 5: Perspektif düzeltmeyi tam çözünürlüklü görüntüye uygula
        warped = self._four_point_transform(src_img, best_quad_original)

        # Sonuçları sisteme geri kaydet (Bu kısım senin kodunla aynı)
        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)

        self.context["src_quad"] = best_quad_original.tolist()
        self.context["output_size"] = [warped.shape[1], warped.shape[0]]

        return build_response(context=self)


if __name__ == '__main__':
    Executor(sys.argv[1]).run()