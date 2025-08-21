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


# -----------------------------------------------------------------------------
# 1. PARAMETRE YÖNETİMİ (BASİTLEŞTİRİLDİ)
# -----------------------------------------------------------------------------
class Params:
    def __init__(self, config=None):
        config = config or {}
        self.resize_longest_edge = config.get("resize_longest_edge", 1200)  # Daha yüksek çözünürlükle çalışalım
        self.score_min_area_ratio = config.get("score_min_area_ratio", 0.15)
        self.approx_poly_epsilon_ratio = config.get("approx_poly_epsilon_ratio", 0.02)
        self.min_confidence_threshold = config.get("min_confidence_threshold",
                                                   0.5)  # Yüksek kaliteli bir sonuç arıyoruz


# -----------------------------------------------------------------------------
# 2. YARDIMCI & GEOMETRİ FONKSİYONLARI
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


# -----------------------------------------------------------------------------
# 3. "EVRENSEL" ÖN İŞLEME ZİNCİRİ
# -----------------------------------------------------------------------------
def universal_preprocess(image: np.ndarray) -> np.ndarray:
    """
    Her türlü ışık ve zemin koşuluna uyum sağlamak için tasarlanmış
    detaylı bir ön işleme zinciri.
    """
    # Adım 1: LAB Renk Uzayında Işık Kanalını (L) Al
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Adım 2: Gölgeleri ve Işık Farklarını Yok Et (Illumination Normalization)
    # Görüntünün genel aydınlatma desenini (çok bulanık versiyonu) bul
    h, w = l_channel.shape
    blur_kernel_size = int(w / 3)
    if blur_kernel_size % 2 == 0: blur_kernel_size += 1
    blurred_l = cv2.GaussianBlur(l_channel, (blur_kernel_size, blur_kernel_size), 0)
    # Orijinal aydınlatmayı çıkararak düzleştir
    normalized_l = cv2.divide(l_channel, blurred_l, scale=255)

    # Adım 3: Yerel Kontrastı Artır (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(normalized_l)

    # Adım 4: Kenarları Koruyarak Gürültü Temizle
    denoised_l = cv2.medianBlur(enhanced_l, 5)

    return denoised_l


# -----------------------------------------------------------------------------
# 4. ANA BİLEŞEN (TEK VE GÜÇLÜ BİR YAPI İLE)
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
        work_img = cv2.resize(src_img_orig, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        # 1. Adım: Evrensel Ön İşlemeyi Uygula
        print("Evrensel ön işleme zinciri çalıştırılıyor...")
        processed_img = universal_preprocess(work_img)

        # 2. Adım: Temiz Görüntüden Kenarları ve Konturları Bul
        edged = cv2.Canny(processed_img, 30, 100)
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
                                  iterations=3)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        document_quad = None
        if contours:
            # Sadece en büyük alanı olan konturu al, çünkü ön işleme sonrası en belirgin o olmalı.
            best_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(best_contour)
            total_area = work_img.shape[0] * work_img.shape[1]

            if area / total_area > self.params.score_min_area_ratio:
                peri = cv2.arcLength(best_contour, True)
                approx = cv2.approxPolyDP(best_contour, self.params.approx_poly_epsilon_ratio * peri, True)

                if len(approx) == 4 and cv2.isContourConvex(approx):
                    print(f"Başarılı: Geçerli bir dörtgen bulundu.")
                    document_quad = approx.reshape(4, 2).astype(np.float32)
                else:
                    print("Uyarı: En büyük kontur bir dörtgen değil.")
            else:
                print("Uyarı: En büyük konturun alanı çok küçük.")
        else:
            print("Uyarı: Ön işleme sonrası hiç kontur bulunamadı.")

        # 3. Adım: Perspektifi Düzelt
        warped = None
        if document_quad is not None:
            document_quad /= scale
            warped = _four_point_transform(src_img_orig, document_quad)

        if warped is None:
            print("Tüm analizler başarısız. Fallback olarak orijinal görüntü kullanılıyor.")
            warped = src_img_orig
            document_quad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)
        self.context["src_quad"] = document_quad.tolist()
        self.context["output_size"] = [warped.shape[1], warped.shape[0]]
        return build_response(context=self)


if __name__ == "__main__":
    Executor(sys.argv[1]).run()