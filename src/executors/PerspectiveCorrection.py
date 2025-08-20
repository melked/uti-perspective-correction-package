import os
import sys
import cv2
import numpy as np
import math
from typing import Optional

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveCorrection.src.utils.response import build_response
from components.PerspectiveCorrection.src.models.PackageModel import PackageModel


# -----------------------------------------------------------------------------
# 1. Geometri ve Yardımcı Fonksiyonlar
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


# -----------------------------------------------------------------------------
# 2. Nihai Tespit Stratejisi: Hibrit Ön İşleme Hunisi
# -----------------------------------------------------------------------------
def find_document_with_hybrid_preprocessing(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Farklı ön işleme tekniklerinin en iyi yanlarını birleştirerek tek ve sağlam bir maske oluşturur,
    ve bu maskeden en olası belge dörtgenini çıkarır.
    """
    orig_h, orig_w = image.shape[:2]

    # Adım 1: Görüntü Standardizasyonu
    scale = 800 / max(orig_h, orig_w)
    resized = cv2.resize(image, (int(orig_w * scale), int(orig_h * scale)))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Adım 2: Çok Kanallı Analiz İçin Girdileri Hazırla
    # "Süper Göz": Aydınlatması düzleştirilmiş görüntü
    kernel_size = int(min(gray.shape[:2]) / 5);
    if kernel_size % 2 == 0: kernel_size += 1
    blurred_bg = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    flattened = cv2.divide(gray, blurred_bg, scale=255)

    # "Normal Göz": Standart bulanıklaştırılmış görüntü
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adım 3: Hibrit Kenar Haritası Oluştur
    # Canny, net detayları yakalar
    canny_edges = cv2.Canny(blurred, 50, 150)
    # Eşikleme, genel silüeti yakalar
    _, flat_thresh = cv2.threshold(flattened, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # İki haritayı birleştirerek en iyi yanlarını al
    hybrid_map = cv2.bitwise_or(canny_edges, flat_thresh)

    # Adım 4: "Süper Morfoloji" ile Maskeyi Mükemmelleştir
    # Spiralli defterler ve yazılar gibi büyük boşlukları doldurmak için büyük bir kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(hybrid_map, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Küçük gürültüleri temizle
    closed = cv2.erode(closed, None, iterations=2)
    closed = cv2.dilate(closed, None, iterations=1)

    # Adım 5: Nihai Köşe Tespiti
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)

    # Güvenlik Kontrolü: Alan çok küçük veya çok büyükse reddet
    area = cv2.contourArea(c)
    total_area = closed.shape[0] * closed.shape[1]
    if not (0.05 < area / total_area < 0.95):
        return None

    # En sağlam köşe bulma yöntemi olarak minAreaRect kullan
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)

    # Köşeleri orijinal görüntü boyutuna geri ölçekle
    box /= scale

    return box.astype(np.float32)


# -----------------------------------------------------------------------------
# Ana Bileşen
# -----------------------------------------------------------------------------
class PerspectiveCorrection(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.context = {}
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")

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
        src_img = self._prepare_image(img_obj.value)
        h, w = src_img.shape[:2]

        document_quad = None

        print("Hibrit Ön İşleme Hunisi stratejisi deneniyor...")
        document_quad = find_document_with_hybrid_preprocessing(src_img)

        if document_quad is not None:
            print("Başarılı: Uygun bir belge adayı bulundu.")
        else:
            print("Hibrit strateji başarısız. Fallback olarak tüm görüntü kullanılıyor.")
            document_quad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

        warped = _four_point_transform(src_img, document_quad)
        if warped is None:
            warped = src_img

        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)
        self.context["src_quad"] = document_quad.tolist()
        self.context["output_size"] = [warped.shape[1], warped.shape[0]]
        return build_response(context=self)


# -----------------------------------------------------------------------------
# Çalıştırıcı
# -----------------------------------------------------------------------------
Executor(sys.argv[1]).run()