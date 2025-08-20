import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveCorrection.src.utils.response import build_response
from components.PerspectiveCorrection.src.models.PackageModel import PackageModel


# -----------------------------------------------------------------------------
# 1. PARAMETRE YÖNETİMİ SINIFI
# -----------------------------------------------------------------------------
class Params:
    """ Algoritmanın kullandığı tüm parametreleri merkezi olarak yönetir. """

    def __init__(self, config=None):
        config = config or {}
        # Ön İşleme Parametreleri
        self.resize_longest_edge = config.get("resize_longest_edge", 1000)
        self.blur_ksize = tuple(config.get("blur_ksize", (5, 5)))

        # Canny Kenar Tespiti Parametreleri
        self.canny_min = config.get("canny_min", 75)
        self.canny_max = config.get("canny_max", 200)

        # goodFeaturesToTrack (Köşe Tespiti) Parametreleri
        self.feature_max_corners = config.get("feature_max_corners", 20)
        self.feature_quality_level = config.get("feature_quality_level", 0.01)
        self.feature_min_distance = config.get("feature_min_distance", 20)

        # Kontur Tabanlı Yedek Yöntem Parametreleri
        self.contour_min_area_ratio = config.get("contour_min_area_ratio", 0.1)
        self.approx_poly_epsilon_ratio = config.get("approx_poly_epsilon_ratio", 0.02)


# -----------------------------------------------------------------------------
# 2. Geometri ve Yardımcı Fonksiyonlar
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
# 3. GITHUB PROJESİNDEN ALINAN ANA TESPİT MANTIĞI
# -----------------------------------------------------------------------------
def find_document_corners_from_repo(image: np.ndarray, params: Params) -> Optional[np.ndarray]:
    """ GitHub projesindeki mantığı kullanarak belge köşelerini bulur. """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, params.blur_ksize, 0)

    # Ana Strateji: goodFeaturesToTrack ile köşe avı
    corners = cv2.goodFeaturesToTrack(
        blurred,
        maxCorners=params.feature_max_corners,
        qualityLevel=params.feature_quality_level,
        minDistance=params.feature_min_distance
    )

    if corners is not None and len(corners) >= 4:
        print("Bilgi: Köşeler 'goodFeaturesToTrack' ile bulundu.")
        corners = np.squeeze(corners)
        # En dıştaki dörtgeni bulmak için Convex Hull kullanmak daha sağlamdır
        hull = cv2.convexHull(corners)
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, params.approx_poly_epsilon_ratio * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)

    # Yedek Strateji: Eğer köşe bulunamazsa, kontur tabanlı tespiti dene
    print("Uyarı: 'goodFeaturesToTrack' başarısız, kontur tabanlı yedek deneniyor...")
    edged = cv2.Canny(blurred, params.canny_min, params.canny_max)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours: return None

    # En büyük 5 konturu dene
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        total_area = image.shape[0] * image.shape[1]
        if cv2.contourArea(c) < total_area * params.contour_min_area_ratio:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, params.approx_poly_epsilon_ratio * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            print("Bilgi: Köşeler kontur tabanlı yedek ile bulundu.")
            return approx.reshape(4, 2).astype(np.float32)

    return None


# -----------------------------------------------------------------------------
# 4. Ana Bileşen
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

        # Görüntüyü standart bir boyuta indirge
        scale = self.params.resize_longest_edge / max(h, w)
        work_img = cv2.resize(src_img_orig, (int(w * scale), int(h * scale)))

        document_quad = None
        warped = None

        print("GitHub projesindeki mantık deneniyor...")
        candidate_quad_scaled = find_document_corners_from_repo(work_img, self.params)

        if candidate_quad_scaled is not None:
            document_quad = candidate_quad_scaled / scale  # Köşeleri orijinal boyuta geri ölçekle
            warped = _four_point_transform(src_img_orig, document_quad)
            if warped is not None:
                print("Başarılı: Belge bulundu ve düzeltildi.")

        if warped is None:
            print("Tespit başarısız. Fallback olarak tüm görüntü kullanılıyor.")
            warped = src_img_orig
            document_quad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)
        self.context["src_quad"] = document_quad.tolist()
        self.context["output_size"] = [warped.shape[1], warped.shape[0]]
        return build_response(context=self)


# -----------------------------------------------------------------------------
# 5. Çalıştırıcı
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    Executor(sys.argv[1]).run()