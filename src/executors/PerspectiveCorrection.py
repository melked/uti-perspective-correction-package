import os
import sys
import cv2
import numpy as np

from typing import Optional, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveCorrection.src.utils.response import build_response
from components.PerspectiveCorrection.src.models.PackageModel import PackageModel


# -----------------------------------------------------------------------------
# 1. Geometri Yardımcı Fonksiyonları (Değişiklik Gerekmiyor)
# -----------------------------------------------------------------------------
# Bu fonksiyonlar standart ve görev için gerekli. Olduğu gibi kalabilirler.

def _order_points(pts: np.ndarray) -> np.ndarray:
    """
    Köşe noktalarını [sol-üst, sağ-üst, sağ-alt, sol-alt] sırasına dizer.
    """
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Sol-üst köşe en küçük toplama sahiptir
    rect[2] = pts[np.argmax(s)]  # Sağ-alt köşe en büyük toplama sahiptir

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Sağ-üst köşe en küçük farka sahiptir
    rect[3] = pts[np.argmax(diff)]  # Sol-alt köşe en büyük farka sahiptir

    return rect


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Verilen 4 köşe noktasına göre görüntünün perspektifini düzeltir.
    """
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect

    # Çıktı görüntüsünün genişliğini hesapla
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    # Çıktı görüntüsünün yüksekliğini hesapla
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    # Hedef köşe noktalarını belirle (düzleştirilmiş görüntü)
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Perspektif dönüşüm matrisini hesapla ve uygula
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LANCZOS4)

    return warped


# -----------------------------------------------------------------------------
# 2. Güçlendirilmiş Tek Adımlı Belge Tespiti
# -----------------------------------------------------------------------------

def find_document_contour(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Bir görüntüdeki en büyük, dört köşeli belge benzeri nesnenin konturunu bulur.

    Bu fonksiyon, sadeleştirilmiş ve robust bir işlem hattı kullanır:
    Gri Tonlama -> Gaussian Blur -> Canny Kenar Tespiti -> Kontur Bulma -> Şekil Yaklaşımı
    """
    # Görüntü alanının %20'sinden küçük konturları göz ardı etmek için bir eşik belirle
    min_area_ratio = 0.2
    img_area = image.shape[0] * image.shape[1]

    # Adım 1: Ön İşleme
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Gürültüyü azaltmak ve dokuyu yumuşatmak için Gaussian Blur uygula
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adım 2: Kenar Tespiti
    # Canny kenar tespiti, belgenin ana hatlarını ortaya çıkarır
    edged = cv2.Canny(blurred, 75, 200)

    # Adım 3: Kontur Bulma
    # Kenar haritasındaki tüm kapalı şekilleri (konturları) bul
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Konturları alana göre büyükten küçüğe sırala
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Adım 4: En Olası Konturu Bul
    # En büyük konturları gezerek 4 köşeli olanı ara
    for c in contours:
        # Alan kontrolü: Çok küçük konturları atla
        if cv2.contourArea(c) < img_area * min_area_ratio:
            break

        peri = cv2.arcLength(c, True)
        # Konturu daha basit bir çokgene yaklaştır (approximate)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # Eğer yaklaştırılan şeklin 4 köşesi varsa, bu bizim belgemizdir
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)

    return None


# -----------------------------------------------------------------------------
# 3. Ana Bileşen
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

        # Tek ve güçlü fonksiyon ile belge köşelerini bul
        document_quad = find_document_contour(src_img)

        # Eğer bir kontur bulunamazsa, fallback olarak tüm görüntüyü kullan
        if document_quad is None:
            print("Belge konturu bulunamadı. Fallback olarak tüm görüntü kullanılıyor.")
            document_quad = np.array([
                [0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]
            ], dtype=np.float32)

        # Perspektifi düzelt
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