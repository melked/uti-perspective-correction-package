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
    """
    PyImageSearch makalesindeki klasik ve güçlü adımları kullanarak
    perspektif düzeltme yapan bileşen.
    """

    class Params:
        """Algoritma için temel parametreler."""

        def __init__(self, config=None):
            config = config or {}
            # Performans için yeniden boyutlandırma yüksekliği
            self.resize_height = config.get("resize_height", 500)
            # Kontur tespiti için minimum alan oranı
            self.min_area_ratio = config.get("min_area_ratio", 0.15)
            # Dörtgen yaklaştırma hassasiyeti
            self.approx_poly_epsilon_ratio = config.get("approx_poly_epsilon_ratio", 0.02)

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
        """4 noktayı sol-üst, sağ-üst, sağ-alt, sol-alt olarak sıralar."""
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
        """Sıralanmış 4 noktayı kullanarak perspektif düzeltme uygular."""
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

    # --- PYIMAGESEARCH AKIŞINI UYGULAYAN METOTLAR ---
    def _preprocess_for_detection(self, image: np.ndarray) -> np.ndarray:
        """Adım 1 & 2: Görüntüyü griye çevir, bulanıklaştır ve kenarları bul."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)
        print("Ön işleme tamamlandı: Gri tonlama -> Gaussian Blur -> Canny Edge")
        return edged

    def _find_document_contour(self, edged_image: np.ndarray) -> Optional[np.ndarray]:
        """Adım 3: Kenar haritasındaki en büyük dörtgeni bulur."""
        # Not: cv2.RETR_EXTERNAL, sadece en dış konturları bularak performansı artırır.
        contours, _ = cv2.findContours(edged_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("Hiç kontur bulunamadı.")
            return None

        # Konturları alana göre büyükten küçüğe sırala
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # En büyük konturları gezerek 4 köşeli olanı ara
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, self.params.approx_poly_epsilon_ratio * peri, True)

            # Eğer konturumuzun 4 köşesi varsa, onu belge olarak kabul et
            if len(approx) == 4:
                print("4 köşeli belge konturu bulundu.")
                return approx.reshape(4, 2).astype(np.float32)

        print("4 köşeli bir kontur bulunamadı.")
        return None

    # --- ANA İŞ AKIŞI: ORKESTRA ŞEFİ ---
    def run(self):
        # 1. Hazırlık: Görüntüyü yükle ve hazırla
        img_obj = Image.get_frame(img=self.image, redis_db=self.redis_db)
        if img_obj is None or img_obj.value is None: raise ValueError("Girdi görüntüsü alınamadı.")

        src_img_orig = img_obj.value
        if src_img_orig.dtype != np.uint8: src_img_orig = cv2.normalize(src_img_orig, None, 0, 255,
                                                                        cv2.NORM_MINMAX).astype(np.uint8)
        if src_img_orig.ndim == 2:
            src_img_orig = cv2.cvtColor(src_img_orig, cv2.COLOR_GRAY2BGR)
        elif src_img_orig.shape[-1] == 4:
            src_img_orig = cv2.cvtColor(src_img_orig, cv2.COLOR_BGRA2BGR)

        # Orijinal en/boy oranını koru
        h_orig, w_orig = src_img_orig.shape[:2]
        ratio = h_orig / self.params.resize_height

        # Performans için görüntüyü yeniden boyutlandır
        work_img = cv2.resize(src_img_orig, (int(w_orig / ratio), self.params.resize_height))

        # 2. Ön İşleme ve Kenar Tespiti
        edged_image = self._preprocess_for_detection(work_img)

        # 3. Belge Konturunu Bulma
        document_quad_scaled = self._find_document_contour(edged_image)

        warped = None
        document_quad_orig = None

        # 4. Perspektif Düzeltme (Eğer kontur bulunduysa)
        if document_quad_scaled is not None:
            # Köşe noktalarını orijinal görüntü boyutuna geri ölçekle
            document_quad_orig = document_quad_scaled * ratio

            # Alan kontrolü yap
            if cv2.contourArea(document_quad_orig) / (w_orig * h_orig) > self.params.min_area_ratio:
                print("Perspektif dönüşümü uygulanıyor...")
                warped = self._four_point_transform(src_img_orig, document_quad_orig)
            else:
                print("Bulunan kontur minimum alan oranının altında kaldı.")

        # 5. Sonuçlandırma (Eğer bir şey ters gittiyse fallback)
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