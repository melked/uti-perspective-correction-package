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
# 1. Geometri, Puanlama ve Yardımcı Fonksiyonlar
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


def _score_candidate(contour: np.ndarray, image_shape: tuple) -> float:
    h, w = image_shape[:2];
    total_area = w * h
    area = cv2.contourArea(contour)
    if not (0.02 < area / total_area < 0.95): return 0.0

    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    if len(approx) != 4 or not cv2.isContourConvex(approx): return 0.0

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0: return 0.0
    solidity = area / hull_area

    M = cv2.moments(approx);
    if M["m00"] == 0: return 0.0
    cx = M["m10"] / M["m00"];
    cy = M["m01"] / M["m00"]
    centrality_score = 1.0 - (np.linalg.norm(np.array([cx, cy]) - np.array([w / 2, h / 2])) / (max(w, h) / 2))

    return (area / total_area * 0.5) + (centrality_score * 0.3) + (solidity * 0.2)


# -----------------------------------------------------------------------------
# Piramidin Katmanları: Stratejiler
# -----------------------------------------------------------------------------

def find_best_candidate_from_pool(image: np.ndarray) -> Optional[np.ndarray]:
    """ ANA STRATEJİ: Farklı maskelerden aday havuzu oluşturur ve en iyisini seçer. """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aydınlatma Normalizasyonu (Sınır Gözcüsü tekniği)
    kernel_size = int(min(gray.shape[:2]) / 5)
    if kernel_size % 2 == 0: kernel_size += 1
    blurred_bg = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    flattened = cv2.divide(gray, blurred_bg, scale=255)

    # 3 Farklı Maske Üretimi
    _, mask_flat = cv2.threshold(flattened, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask_canny = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
    mask_adaptive = cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (7, 7), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 21, 5)

    # Aday Havuzunu Oluştur
    candidate_pool = []
    for mask in [mask_flat, mask_canny, mask_adaptive]:
        # Gürültüyü temizle ve delikleri kapat
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        candidate_pool.extend(contours)

    if not candidate_pool: return None

    # Havuzdaki en iyi adayı bul
    best_candidate, best_score = None, 0.3  # Minimum geçme notu
    for c in sorted(candidate_pool, key=cv2.contourArea, reverse=True)[:30]:  # En büyük 30 adayı değerlendir
        score = _score_candidate(c, image.shape)
        if score > best_score:
            best_score = score
            peri = cv2.arcLength(c, True)
            best_candidate = cv2.approxPolyDP(c, 0.02 * peri, True)

    return best_candidate.reshape(4, 2).astype(np.float32) if best_candidate is not None else None


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
        src_img_orig = self._prepare_image(img_obj.value)
        h, w = src_img_orig.shape[:2]

        # --- PİRAMİT STRATEJİSİ DEVREDE ---

        # KATMAN 1: Görüntü Standardizasyonu
        scale = 1000 / max(h, w)  # Dinamik boyutlandırma
        resized_img = cv2.resize(src_img_orig, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        # Keskinleştirme
        sharpened_img = cv2.addWeighted(resized_img, 1.8, cv2.GaussianBlur(resized_img, (0, 0), 3), -0.8, 0)

        document_quad = None

        # KATMAN 2: Akıllı Aday Tespiti
        print("Piramidin ana katmanı (Akıllı Aday Tespiti) deneniyor...")
        document_quad = find_best_candidate_from_pool(sharpened_img)

        # KATMAN 3: Özel Durumlar (Gerekirse Hough, K-Means vb. buraya eklenebilir)

        if document_quad is not None:
            document_quad /= scale  # Köşeleri orijinal görüntü boyutuna geri getir
            print("Başarılı: Uygun bir belge adayı bulundu.")
        else:
            print("Tüm katmanlar başarısız. Fallback olarak tüm görüntü kullanılıyor.")
            document_quad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

        warped = _four_point_transform(src_img_orig, document_quad)
        if warped is None:
            warped = src_img_orig

        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)
        self.context["src_quad"] = document_quad.tolist()
        self.context["output_size"] = [warped.shape[1], warped.shape[0]]
        return build_response(context=self)


# -----------------------------------------------------------------------------
# Çalıştırıcı
# -----------------------------------------------------------------------------
Executor(sys.argv[1]).run()