import os
import sys
import cv2
import numpy as np

from typing import Optional, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveTransformation.src.utils.response import build_response
from components.PerspectiveTransformation.src.models.PackageModel import PackageModel


# -----------------------------------------------------------------------------
# 1. Geometri Yardımcı Fonksiyonları (Değişiklik Yok)
# -----------------------------------------------------------------------------
def _order_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LANCZOS4)
    return warped


# -----------------------------------------------------------------------------
# 2. SEVİYE 3: Gelişmiş Belge Tespiti (En Zor Durumlar İçin)
# -----------------------------------------------------------------------------

def _preprocess_for_contours_advanced(image: np.ndarray) -> np.ndarray:
    """
    Görüntüyü en zorlu senaryolar için hazırlar: düşük kontrast, yansımalar ve gölgeler.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Adım 1: Bilateral Filtre ile yüzeydeki parazitleri (yansıma vb.) azalt ama kenarları koru.
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)

    # Adım 2: Adaptif Eşikleme ile lokal kontrastı ortaya çıkar. Gölgelerle başa çıkmada çok etkilidir.
    thresh = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 4)

    # Adım 3: Agresif Morfolojik Kapatma.
    # Belge üzerindeki metinleri ve kenarlardaki boşlukları birleştirerek tek bir kapalı alan oluşturur.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Adım 4: Kalan küçük gürültüleri temizle.
    closed = cv2.erode(closed, None, iterations=2)
    closed = cv2.dilate(closed, None, iterations=2)

    return closed


def find_document_contour_final(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Gelişmiş ön işleme hattını kullanarak en zorlu görüntülerde bile belge konturunu bulur.
    """
    preprocessed = _preprocess_for_contours_advanced(image)

    contours, _ = cv2.findContours(preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # En büyük konturun 4 köşesi olup olmadığını kontrol et
    if len(contours) > 0:
        c = contours[0]
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # En büyük kontur bir dörtgen ise onu döndür
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype(np.float32)

    return None


# -----------------------------------------------------------------------------
# 3. Ana Bileşen (Nihai Fonksiyonu Kullanacak Şekilde)
# -----------------------------------------------------------------------------

class PerspectiveTransformation(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.context = {}
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

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

    def run(self):
        img_obj = Image.get_frame(img=self.image, redis_db=self.redis_db)
        if img_obj is None or img_obj.value is None:
            raise ValueError("No input image provided or failed to load.")

        src_img = self._prepare_image(img_obj.value)
        h, w = src_img.shape[:2]

        # En gelişmiş ve sağlam fonksiyonu çağır
        document_quad = find_document_contour_final(src_img)

        if document_quad is None:
            print("Belge konturu bulunamadı. Fallback olarak tüm görüntü kullanılıyor.")
            document_quad = np.array([
                [0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]
            ], dtype=np.float32)

        warped = _four_point_transform(src_img, document_quad)

        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)

        self.context["src_quad"] = document_quad.tolist()
        self.context["output_size"] = [warped.shape[1], warped.shape[0]]

        return build_response(context=self)


# -----------------------------------------------------------------------------
# 4. Çalıştırıcı
# -----------------------------------------------------------------------------
Executor(sys.argv[1]).run()