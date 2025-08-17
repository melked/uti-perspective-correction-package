import os
import sys
import cv2
import numpy as np
import math

from typing import Optional, Tuple, List

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

# ... (diğer importlar ve class tanımları aynı) ...
from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveTransformation.src.utils.response import build_response
from components.PerspectiveTransformation.src.models.PackageModel import PackageModel

def _order_points(pts: np.ndarray) -> np.ndarray:
    # ... (kod aynı)
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
    # ... (kod aynı)
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LANCZOS4)
    return warped


def _preprocess_for_contours_advanced(image: np.ndarray) -> np.ndarray:
    # ... (Bu ön işleme fonksiyonu iyi, aynı kalabilir) ...
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    closed = cv2.erode(closed, None, iterations=2)
    closed = cv2.dilate(closed, None, iterations=2)
    return closed


def find_document_contour_with_scoring(image: np.ndarray) -> Optional[np.ndarray]:
    """
    En büyük 10 adayı bulur ve onları akıllı kriterlere göre puanlayarak en iyisini seçer.
    """
    preprocessed = _preprocess_for_contours_advanced(image)
    contours, _ = cv2.findContours(preprocessed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Konturları alana göre sırala ve en büyük 10 tanesini al
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    h, w = image.shape[:2]
    img_center = np.array([w / 2, h / 2])

    best_score = -1
    best_quad = None

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # Sadece dörtgen adayları değerlendir
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad = approx.reshape(4, 2).astype(np.float32)

            # --- PUANLAMA KRİTERLERİ ---
            # 1. Alan Puanı (Büyük olan iyidir)
            area = cv2.contourArea(quad)
            area_score = np.clip(area / (w * h), 0, 1)

            # 2. Merkezilik Puanı (Merkeze yakın olan iyidir)
            M = cv2.moments(quad)
            if M["m00"] == 0: continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            dist = np.linalg.norm(np.array([cx, cy]) - img_center)
            centrality_score = 1 - np.clip(dist / (max(w, h) / 2), 0, 1)

            # 3. En-Boy Oranı Puanı (Belgeye benzeyen iyidir)
            rect = _order_points(quad)
            (tl, tr, br, bl) = rect
            width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
            height = (np.linalg.norm(tl - bl) + np.linalg.norm(tr - br)) / 2
            if min(width, height) < 1e-6: continue
            aspect_ratio = max(width, height) / min(width, height)
            # A4 (1.41) veya benzeri oranları tercih et, aşırı oranları cezalandır
            aspect_score = math.exp(-0.5 * ((aspect_ratio - 1.4) ** 2))

            # --- FİNAL SKOR ---
            # Kriterleri ağırlıklandırarak topla
            final_score = (area_score * 0.4) + (centrality_score * 0.4) + (aspect_score * 0.2)

            if final_score > best_score:
                best_score = final_score
                best_quad = quad

    return best_quad

class PerspectiveTransformation(Component):
    # __init__, bootstrap, _prepare_image metodları aynı...
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
        # ... (görüntü yükleme kısmı aynı) ...
        if img_obj is None or img_obj.value is None:
            raise ValueError("No input image provided or failed to load.")

        src_img = self._prepare_image(img_obj.value)
        h, w = src_img.shape[:2]

        print("Akıllı puanlama stratejisi deneniyor...")
        document_quad = find_document_contour_with_scoring(src_img)

        # Fallback: Puanlama sonucu bir şey bulunamazsa tüm görüntüyü kullan
        if document_quad is None:
            print("Uygun bir aday bulunamadı. Fallback olarak tüm görüntü kullanılıyor.")
            document_quad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

        warped = _four_point_transform(src_img, document_quad)
        # ... (geri kalan kodlar aynı) ...
        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)

        self.context["src_quad"] = document_quad.tolist()
        self.context["output_size"] = [warped.shape[1], warped.shape[0]]

        return build_response(context=self)

Executor(sys.argv[1]).run()