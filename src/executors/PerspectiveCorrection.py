import os
import sys
import cv2
import numpy as np
import math
from collections import defaultdict
from typing import Optional, List, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveCorrection.src.utils.response import build_response
from components.PerspectiveCorrection.src.models.PackageModel import PackageModel


# -----------------------------------------------------------------------------
# 1. Geometri ve YENİ Gelişmiş Puanlama Fonksiyonları
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
    if not (0.03 < area / total_area < 0.95): return 0.0

    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    if len(approx) != 4 or not cv2.isContourConvex(approx): return 0.0

    # YENİLİK: Sağlamlık (Solidity) Puanı
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0: return 0.0
    solidity = area / hull_area

    M = cv2.moments(approx);
    if M["m00"] == 0: return 0.0
    cx = M["m10"] / M["m00"];
    cy = M["m01"] / M["m00"]
    centrality_score = 1.0 - (np.linalg.norm(np.array([cx, cy]) - np.array([w / 2, h / 2])) / (max(w, h) / 2))

    # YENİLİK: Esnek En-Boy Oranı Puanı
    rect = _order_points(approx.reshape(4, 2))
    (tl, tr, br, bl) = rect
    width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    height = (np.linalg.norm(tl - bl) + np.linalg.norm(tr - br)) / 2
    if min(width, height) < 1: return 0.0
    aspect_ratio = max(width, height) / min(width, height)
    if 1.1 < aspect_ratio < 1.9:
        aspect_score = 1.0  # Kitap, A4, planlayıcı gibi oranlar için tam puan
    else:
        aspect_score = 0.5  # Diğer oranlar için yarım puan

    return (solidity * 0.4) + (centrality_score * 0.4) + (aspect_score * 0.2)


# -----------------------------------------------------------------------------
# UZMAN STRATEJİLERİ (v2.0)
# -----------------------------------------------------------------------------
def find_best_quad_from_contours(contours: list, image_shape: tuple) -> Optional[np.ndarray]:
    if not contours: return None
    best_quad, best_score = None, 0.3  # Minimum geçme notu artırıldı
    # En büyük 10 adayı değerlendir
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        score = _score_candidate(c, image_shape)
        if score > best_score:
            best_score = score
            peri = cv2.arcLength(c, True)
            best_quad = cv2.approxPolyDP(c, 0.02 * peri, True)
    return best_quad.reshape(4, 2).astype(np.float32) if best_quad is not None else None


def stage1_fast_and_simple(image: np.ndarray) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return find_best_quad_from_contours(contours, image.shape)


def stage2_boundary_watcher(image: np.ndarray) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel_size = int(min(image.shape[:2]) / 5)
    if kernel_size % 2 == 0: kernel_size += 1
    blurred_bg = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    flattened = cv2.divide(gray, blurred_bg, scale=255)
    _, thresh = cv2.threshold(flattened, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # YENİLİK: Gürültü temizleme adımı
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=1)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=3)
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return find_best_quad_from_contours(contours, image.shape)


def stage3_content_analyzer(image: np.ndarray) -> Optional[np.ndarray]:
    h, w = image.shape[:2]
    scale = 300 / max(h, w)
    small_img = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    pixels = small_img.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = centers.astype(np.uint8)
    lab_centers = cv2.cvtColor(centers.reshape(1, -1, 3), cv2.COLOR_BGR2LAB)[0]

    # YENİLİK: En parlak iki kümeyi birleştirmeyi dene
    sorted_clusters = sorted(range(len(centers)), key=lambda i: lab_centers[i][0], reverse=True)

    for cluster_indices in [[sorted_clusters[0]], [sorted_clusters[0], sorted_clusters[1]]]:
        mask = np.zeros(labels.shape[:2], dtype=np.uint8)
        for idx in cluster_indices:
            mask[labels.flatten() == idx] = 255

        mask = mask.reshape(small_img.shape[:2])
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        quad = find_best_quad_from_contours([closed], image.shape)
        if quad is not None: return quad  # Başarılı bir sonuç bulunursa hemen döndür

    return None


# -----------------------------------------------------------------------------
# Ana Bileşen ve Çalıştırıcı (Değişiklik Yok)
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

        document_quad = None
        # Not: Spiralli defter için Stage 4'ü de ekleyebiliriz, ancak ilk 3'ü güçlendirmek öncelikliydi.
        strategies = {
            "Hızlı Gözcü": stage1_fast_and_simple,
            "Sınır Gözcüsü": stage2_boundary_watcher,
            "İçerik Analisti": stage3_content_analyzer,
        }

        for name, strategy in strategies.items():
            print(f"Aşama ( {name} ) deneniyor...")
            document_quad = strategy(src_img)
            if document_quad is not None:
                print(f"Başarılı: Belge '{name}' stratejisi ile bulundu.")
                break

        if document_quad is None:
            print("Tüm uzmanlar başarısız. Fallback olarak tüm görüntü kullanılıyor.")
            h, w = src_img.shape[:2]
            document_quad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

        warped = _four_point_transform(src_img, document_quad)
        if warped is None:
            warped = src_img

        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)
        self.context["src_quad"] = document_quad.tolist()
        self.context["output_size"] = [warped.shape[1], warped.shape[0]]
        return build_response(context=self)


Executor(sys.argv[1]).run()