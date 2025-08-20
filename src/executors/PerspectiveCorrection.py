import os
import sys
import cv2
import numpy as np
import math
from collections import defaultdict
from typing import Optional, List, Tuple
from itertools import combinations

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


def _refine_corners(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    winSize = (11, 11);
    zeroZone = (-1, -1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    return cv2.cornerSubPix(gray, corners.astype(np.float32), winSize, zeroZone, criteria)


# -----------------------------------------------------------------------------
# 2. Aday Üretici Stratejiler
# -----------------------------------------------------------------------------
def get_candidates_from_canny(gray_image: np.ndarray) -> List[np.ndarray]:
    """ Canny kenar haritasından aday konturlar üretir. """
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    v = np.median(blurred);
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v));
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(blurred, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def get_candidates_from_thresh(gray_image: np.ndarray) -> List[np.ndarray]:
    """ Arka planı silinmiş görüntüden aday konturlar üretir. """
    kernel_size = int(min(gray_image.shape[:2]) / 5)
    if kernel_size % 2 == 0: kernel_size += 1
    blurred_bg = cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), 0)
    flattened = cv2.divide(gray_image, blurred_bg, scale=255)
    _, thresh = cv2.threshold(flattened, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours


# -----------------------------------------------------------------------------
# 3. "Süper Yargıç": Nihai Puanlama Fonksiyonu
# -----------------------------------------------------------------------------
def score_and_select_best_quad(candidates: list, gray_image: np.ndarray) -> Optional[np.ndarray]:
    if not candidates: return None

    best_quad, best_score = None, 0.3

    # Harris köşe yanıt haritasını önceden hesapla
    harris_response = cv2.cornerHarris(gray_image, 2, 3, 0.04)

    for c in sorted(candidates, key=cv2.contourArea, reverse=True)[:50]:
        h, w = gray_image.shape[:2];
        total_area = w * h
        area = cv2.contourArea(c)
        if not (0.02 < area / total_area < 0.95): continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx): continue

        # Puanlama metrikleri
        area_score = area / total_area

        M = cv2.moments(approx);
        if M["m00"] == 0: continue
        cx = M["m10"] / M["m00"];
        cy = M["m01"] / M["m00"]
        centrality_score = 1.0 - (np.linalg.norm(np.array([cx, cy]) - np.array([w / 2, h / 2])) / (max(w, h) / 2))

        # YENİ: Köşe Keskinliği Puanı
        corner_sharpness_score = 0
        corners = approx.reshape(4, 2)
        for x, y in corners:
            x, y = int(x), int(y)
            # Köşenin etrafındaki küçük bir alandaki Harris yanıtlarının ortalamasını al
            patch = harris_response[max(0, y - 5):y + 5, max(0, x - 5):x + 5]
            if patch.size > 0:
                corner_sharpness_score += patch.mean()
        corner_sharpness_score /= 4  # Ortalamayı al

        # Final Puanı
        final_score = (area_score * 0.4) + (centrality_score * 0.3) + (corner_sharpness_score * 0.3)

        if final_score > best_score:
            best_score = final_score
            best_quad = approx

    return best_quad.reshape(4, 2).astype(np.float32) if best_quad is not None else None


# -----------------------------------------------------------------------------
# 4. Ana Bileşen
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

        # Ön Hazırlık
        scale = 1000 / max(h, w) if max(h, w) > 1000 else 1
        work_img = cv2.resize(src_img_orig, (int(w * scale), int(h * scale)))
        gray_work_img = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)

        # --- Kolektif Akıl Stratejisi ---

        # 1. Tüm uzmanlar aday havuzunu doldurur
        print("Tüm adaylar toplanıyor...")
        candidate_pool = []
        candidate_pool.extend(get_candidates_from_canny(gray_work_img))
        candidate_pool.extend(get_candidates_from_thresh(gray_work_img))

        document_quad = None
        warped = None

        if candidate_pool:
            # 2. "Süper Yargıç" en iyi adayı seçer
            print(f"{len(candidate_pool)} aday bulundu. En iyisi seçiliyor...")
            kaba_quad = score_and_select_best_quad(candidate_pool, gray_work_img)

            if kaba_quad is not None:
                # 3. Son dokunuş: Hassas Ayar
                print("En iyi aday hassaslaştırılıyor...")
                refined_quad = _refine_corners(work_img, kaba_quad)

                document_quad = refined_quad / scale

                warped = _four_point_transform(src_img_orig, document_quad)
                if warped is not None:
                    print("Başarılı: En iyi aday bulundu, hassaslaştırıldı ve doğrulandı.")

        if warped is None:
            print("Tüm uzmanlar başarısız veya geçerli aday bulunamadı. Fallback kullanılıyor.")
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