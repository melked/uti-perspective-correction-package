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


def _line_intersection(line1, line2):
    rho1, theta1 = line1;
    rho2, theta2 = line2
    A = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
    b = np.array([[rho1], [rho2]])
    try:
        x0, y0 = np.linalg.solve(A, b)
        return [int(round(x0)), int(round(y0))]
    except np.linalg.LinAlgError:
        return None


def _score_candidate(contour: np.ndarray, image_shape: tuple) -> float:
    h, w = image_shape[:2];
    total_area = w * h
    area = cv2.contourArea(contour)
    if not (0.03 < area / total_area < 0.95): return 0.0
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    if len(approx) != 4 or not cv2.isContourConvex(approx): return 0.0
    return area / total_area


def find_best_quad_from_contours(contours: list, image_shape: tuple) -> Optional[np.ndarray]:
    if not contours: return None
    best_quad, best_score = None, 0.2
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            score = _score_candidate(approx, image_shape)
            if score > best_score:
                best_score = score
                best_quad = approx
    return best_quad.reshape(4, 2).astype(np.float32) if best_quad is not None else None


# -----------------------------------------------------------------------------
# UZMAN STRATEJİLERİ (Stage 4 Yeniden Yazıldı)
# -----------------------------------------------------------------------------
def stage1_fast_and_simple(image: np.ndarray) -> Optional[np.ndarray]:
    # ... (kod aynı)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return find_best_quad_from_contours(contours, image.shape)


def stage2_boundary_watcher(image: np.ndarray) -> Optional[np.ndarray]:
    # ... (kod aynı)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel_size = int(min(image.shape[:2]) / 5)
    if kernel_size % 2 == 0: kernel_size += 1
    blurred_bg = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    flattened = cv2.divide(gray, blurred_bg, scale=255)
    _, thresh = cv2.threshold(flattened, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return find_best_quad_from_contours(contours, image.shape)


def stage3_content_analyzer(image: np.ndarray) -> Optional[np.ndarray]:
    # ... (kod aynı)
    h, w = image.shape[:2]
    scale = 300 / max(h, w)
    small_img = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    pixels = small_img.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = centers.astype(np.uint8)
    lab_centers = cv2.cvtColor(centers.reshape(1, -1, 3), cv2.COLOR_BGR2LAB)[0]
    brightest_idx = np.argmax([c[0] for c in lab_centers])
    mask = (labels.reshape(small_img.shape[:2]) == brightest_idx).astype(np.uint8) * 255
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) / (w * h) < 0.05: return None
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    return box.astype(np.float32)


def stage4_line_reconstructor(image: np.ndarray) -> Optional[np.ndarray]:  # <<< TAMAMEN YENİLENDİ
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, int(min(image.shape[:2]) / 5))
    if lines is None: return None

    h_lines, v_lines = [], []
    for line in lines:
        rho, theta = line[0]
        if theta < np.pi / 4 or theta > 3 * np.pi / 4:
            v_lines.append((rho, theta))
        else:
            h_lines.append((rho, theta))

    if len(h_lines) < 2 or len(v_lines) < 2: return None

    best_quad, best_score = None, 0.1
    # Tüm olası (2 yatay + 2 dikey) dörtgen kombinasyonlarını dene
    for h1, h2 in combinations(h_lines, 2):
        for v1, v2 in combinations(v_lines, 2):
            p1 = _line_intersection(h1, v1)
            p2 = _line_intersection(h1, v2)
            p3 = _line_intersection(h2, v2)
            p4 = _line_intersection(h2, v1)

            corners = [p for p in [p1, p2, p3, p4] if p is not None]
            if len(corners) == 4:
                quad = np.array(corners, dtype=np.float32)
                # Oluşturulan bu dörtgeni akıllı puanlama sistemimizden geçir
                score = _score_candidate(quad, image.shape)
                if score > best_score:
                    best_score = score
                    best_quad = quad

    return best_quad


# -----------------------------------------------------------------------------
# Ana Bileşen
# -----------------------------------------------------------------------------
class PerspectiveCorrection(Component):
    # ... (sınıfın geri kalanı tamamen aynı)
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
        warped = None
        strategies = {
            "Hızlı Gözcü": stage1_fast_and_simple,
            "Sınır Gözcüsü": stage2_boundary_watcher,
            "İçerik Analisti": stage3_content_analyzer,
            "Çizgi Dedektifi": stage4_line_reconstructor,
        }
        for name, strategy in strategies.items():
            print(f"Aşama ( {name} ) deneniyor...")
            candidate_quad = strategy(src_img)
            if candidate_quad is not None:
                warped_candidate = _four_point_transform(src_img, candidate_quad)
                if warped_candidate is not None:
                    print(f"Başarılı: Belge '{name}' stratejisi ile bulundu ve doğrulandı.")
                    document_quad = candidate_quad
                    warped = warped_candidate
                    break
                else:
                    print(f"Uyarı: '{name}' adayı buldu ancak geometrisi bozuk. Reddediliyor.")
        if warped is None:
            print("Tüm uzmanlar başarısız. Fallback olarak tüm görüntü kullanılıyor.")
            h, w = src_img.shape[:2]
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