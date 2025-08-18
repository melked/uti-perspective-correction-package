import os
import sys
import cv2
import numpy as np
import math
from collections import defaultdict
from typing import Optional

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveCorrection.src.utils.response import build_response
from components.PerspectiveCorrection.src.models.PackageModel import PackageModel


# -----------------------------------------------------------------------------
# 1. Geometri Yardımcı Fonksiyonları
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
    if maxWidth <= 10 or maxHeight <= 10: return None  # Çok küçük çıktıları engelle
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LANCZOS4)


def _score_candidate(quad: np.ndarray, image_shape: tuple) -> float:
    """ Aday bir dörtgeni puanlayan merkezi bir fonksiyon. """
    h, w = image_shape[:2]
    area = cv2.contourArea(quad)

    # GÜVENLİK FİLTRESİ 1: Alan kontrolü
    total_area = w * h
    if not (0.05 < area / total_area < 0.95):
        return 0.0  # Çok küçük veya çok büyük adayları doğrudan ele

    # Puanlama Kriterleri
    area_score = area / total_area

    M = cv2.moments(quad)
    if M["m00"] == 0: return 0.0
    cx = M["m10"] / M["m00"];
    cy = M["m01"] / M["m00"]
    centrality_score = 1.0 - (np.linalg.norm(np.array([cx, cy]) - np.array([w / 2, h / 2])) / (max(w, h) / 2))

    rect = _order_points(quad)
    (tl, tr, br, bl) = rect
    width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    height = (np.linalg.norm(tl - bl) + np.linalg.norm(tr - br)) / 2
    if min(width, height) < 1: return 0.0
    aspect_ratio = max(width, height) / min(width, height)
    aspect_score = math.exp(-0.5 * ((aspect_ratio - 1.4) ** 2))  # A4 (1.41) oranını ödüllendir

    return (area_score * 0.5) + (centrality_score * 0.3) + (aspect_score * 0.2)


# -----------------------------------------------------------------------------
# UZMAN STRATEJİLERİ (İyileştirilmiş)
# -----------------------------------------------------------------------------

def stage1_fast_and_simple(image: np.ndarray) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_quad, best_score = None, 0.0
    if contours:
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                quad = approx.reshape(4, 2).astype(np.float32)
                score = _score_candidate(quad, image.shape)
                if score > best_score:
                    best_score, best_quad = score, quad
    return best_quad if best_score > 0.2 else None  # Minimum puan eşiği


def stage2_boundary_watcher(image: np.ndarray) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel_size = int(min(image.shape[:2]) / 4)  # Daha güçlü bir etki için kernel büyütüldü
    if kernel_size % 2 == 0: kernel_size += 1
    blurred_bg = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    flattened = cv2.divide(gray, blurred_bg, scale=255)
    _, thresh = cv2.threshold(flattened, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_quad, best_score = None, 0.0
    if contours:
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                quad = approx.reshape(4, 2).astype(np.float32)
                score = _score_candidate(quad, image.shape)
                if score > best_score:
                    best_score, best_quad = score, quad
    return best_quad if best_score > 0.2 else None


def stage3_content_analyzer(image: np.ndarray) -> Optional[np.ndarray]:
    h, w = image.shape[:2]
    scale = 300 / max(h, w)  # Daha hızlı işlem için küçültme oranı ayarlandı
    small_img = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    pixels = small_img.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)  # 3 renk kümesi yeterli
    centers = centers.astype(np.uint8)
    lab_centers = cv2.cvtColor(centers.reshape(1, -1, 3), cv2.COLOR_BGR2LAB)[0]
    counts = np.bincount(labels.flatten())
    best_cluster_idx = -1;
    max_score = -1
    # En parlak ve en büyük alanı kaplayan kümeyi bul
    brightest_idx = np.argmax([c[0] for c in lab_centers])
    largest_idx = np.argmax(counts)
    # Eğer en parlak ve en büyük aynı değilse ve en parlak olan çok küçük değilse onu seç
    if brightest_idx != largest_idx and counts[brightest_idx] / pixels.shape[0] > 0.1:
        best_cluster_idx = brightest_idx
    else:
        best_cluster_idx = largest_idx

    mask = (labels.reshape(small_img.shape[:2]) == best_cluster_idx).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    c = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    # GÜVENLİK FİLTRESİ 2: Alan kontrolü
    if not (0.05 < cv2.contourArea(box) / (w * h) < 0.95): return None
    return box.astype(np.float32)


# UZMAN 4: Çizgi Dedektifi (Hough) - Bu uzman çok deneysel ve riskli olduğu için şimdilik kaldırıldı.
# Diğer 3 uzmanın iyileştirilmesiyle çoğu durumun çözülmesi hedeflenmiştir.

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
        strategies = {
            "Hızlı Gözcü": stage1_fast_and_simple,
            "Sınır Gözcüsü": stage2_boundary_watcher,
            "İçerik Analisti": stage3_content_analyzer
        }

        for name, strategy in strategies.items():
            print(f"Aşama ( {name} ) deneniyor...")
            document_quad = strategy(src_img)
            if document_quad is not None:
                print(f"Başarılı: Belge '{name}' stratejisi ile bulundu.")
                break

        if document_quad is None:
            print("Tüm uzmanlar başarısız. Fallback olarak tüm görüntü kullanılıyor.")
            document_quad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

        warped = _four_point_transform(src_img, document_quad)
        if warped is None:
            print("Dönüşüm hatası, fallback kullanılıyor.")
            warped = src_img
            document_quad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)
        self.context["src_quad"] = document_quad.tolist()
        self.context["output_size"] = [warped.shape[1], warped.shape[0]]
        return build_response(context=self)


# -----------------------------------------------------------------------------
# Çalıştırıcı
# -----------------------------------------------------------------------------
Executor(sys.argv[1]).run()