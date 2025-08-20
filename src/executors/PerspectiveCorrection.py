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


def _unsharp_mask(image: np.ndarray, strength: float = 2.0, kernel_size: tuple = (5, 5)) -> np.ndarray:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    blurred = cv2.GaussianBlur(gray, kernel_size, 0)
    sharpened = cv2.addWeighted(gray, 1.0 + strength, blurred, -strength, 0)
    if image.ndim == 3: return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    return sharpened


def _score_candidate(contour: np.ndarray, image_shape: tuple) -> float:
    h, w = image_shape[:2];
    total_area = w * h
    area = cv2.contourArea(contour)
    if not (0.05 < area / total_area < 0.95): return 0.0
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    if len(approx) != 4 or not cv2.isContourConvex(approx): return 0.0
    M = cv2.moments(approx);
    if M["m00"] == 0: return 0.0
    cx = M["m10"] / M["m00"];
    cy = M["m01"] / M["m00"]
    centrality_score = 1.0 - (np.linalg.norm(np.array([cx, cy]) - np.array([w / 2, h / 2])) / (max(w, h) / 2))
    return (area / total_area) * 0.7 + centrality_score * 0.3


def find_best_quad_from_contours(contours: list, image_shape: tuple) -> Optional[np.ndarray]:
    if not contours: return None
    best_quad, best_score = None, 0.25
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:7]:
        score = _score_candidate(c, image_shape)
        if score > best_score:
            best_score = score
            peri = cv2.arcLength(c, True)
            best_quad = cv2.approxPolyDP(c, 0.02 * peri, True)
    return best_quad.reshape(4, 2).astype(np.float32) if best_quad is not None else None


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


# -----------------------------------------------------------------------------
# UZMAN STRATEJİLERİ
# -----------------------------------------------------------------------------
def stage1_fast_and_simple(image: np.ndarray) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return find_best_quad_from_contours(contours, image.shape)


def stage2_boundary_watcher(image: np.ndarray) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel_size = int(min(image.shape[:2]) / 4)
    if kernel_size % 2 == 0: kernel_size += 1
    blurred_bg = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    flattened = cv2.divide(gray, blurred_bg, scale=255)
    _, thresh = cv2.threshold(flattened, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=5)
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return find_best_quad_from_contours(contours, image.shape)


def stage3_line_reconstructor(image: np.ndarray) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, int(min(image.shape[:2]) / 4))
    if lines is None: return None
    h_lines, v_lines = [], []
    for line in lines:
        rho, theta = line[0]
        if theta < np.pi / 4 or theta > 3 * np.pi / 4:
            v_lines.append((rho, theta))
        else:
            h_lines.append((rho, theta))
    if len(h_lines) < 2 or len(v_lines) < 2: return None

    # En dıştaki çizgileri bul (laptop gibi gürültüleri elemek için daha sağlam mantık)
    h_lines.sort(key=lambda x: x[0]);
    v_lines.sort(key=lambda x: x[0])
    top_line = h_lines[0];
    bottom_line = h_lines[-1]
    left_line = v_lines[0];
    right_line = v_lines[-1]

    # Kesişim noktalarını bul
    p1 = _line_intersection(top_line, left_line)
    p2 = _line_intersection(top_line, right_line)
    p3 = _line_intersection(bottom_line, right_line)
    p4 = _line_intersection(bottom_line, left_line)

    corners = [p for p in [p1, p2, p3, p4] if p is not None]
    if len(corners) == 4:
        quad = np.array(corners, dtype=np.float32)
        return quad
    return None


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

        print("Görüntü bulanıklığa karşı keskinleştiriliyor...")
        src_img = _unsharp_mask(src_img_orig)

        h, w = src_img.shape[:2]
        document_quad = None
        warped = None

        strategies = {
            "Hızlı Gözcü": stage1_fast_and_simple,
            "Sınır Gözcüsü": stage2_boundary_watcher,
            "Çizgi Dedektifi": stage3_line_reconstructor,
        }

        for name, strategy in strategies.items():
            print(f"Aşama ( {name} ) deneniyor...")
            candidate_quad = strategy(src_img)

            if candidate_quad is not None:
                warped_candidate = _four_point_transform(src_img_orig, candidate_quad)
                if warped_candidate is not None:
                    print(f"Başarılı: Belge '{name}' stratejisi ile bulundu.")
                    document_quad = candidate_quad
                    warped = warped_candidate
                    break
                else:
                    print(f"Uyarı: '{name}' adayı buldu ancak geometrisi bozuk. Reddediliyor.")

        if warped is None:
            print("Tüm uzmanlar başarısız. Fallback olarak tüm görüntü kullanılıyor.")
            warped = src_img_orig
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