import os
import sys
import cv2
import numpy as np
import math
from collections import defaultdict
from typing import Optional, Tuple, List

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveCorrection.src.utils.response import build_response
from components.PerspectiveCorrection.src.models.PackageModel import PackageModel


# -----------------------------------------------------------------------------
# 1. Geometri Yardımcı Fonksiyonları (Değişiklik Yok)
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
    if maxWidth == 0 or maxHeight == 0: return None
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


# -----------------------------------------------------------------------------
# Stratejiler (Aşama 1, 3, 4 Değişiklik Yok, Aşama 2 Güçlendirildi)
# -----------------------------------------------------------------------------
def stage1_simple_contour(image: np.ndarray) -> Optional[np.ndarray]:
    # ... (Bu fonksiyon aynı)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        for c in contours:
            if cv2.contourArea(c) < (image.shape[0] * image.shape[1] * 0.1): continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                return approx.reshape(4, 2).astype(np.float32)
    return None


def stage2_scored_contour(image: np.ndarray) -> Optional[np.ndarray]:  # <<< BU FONKSİYON GÜÇLENDİRİLDİ
    """
    Ön işleme adımı, iç detayları birleştirmek için daha agresif hale getirildi.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Gürültüyü azaltırken kenarları koru
    bilateral = cv2.bilateralFilter(gray, 11, 75, 75)
    # Adaptif eşikleme
    thresh = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 7)

    # <<< DEĞİŞİKLİK: Kernel boyutu ve tekrar sayısı artırılarak morfolojik kapatma güçlendirildi.
    # Bu, yazı gibi iç detayları belgenin geneliyle birleştirmeye yardımcı olur.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 10))  # Yatay birleştirmeye öncelik ver
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=5)

    # Puanlama kısmı aynı kalıyor, çünkü mantığı doğru. Sadece girdisini iyileştirdik.
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    h, w = image.shape[:2];
    img_center = np.array([w / 2, h / 2])
    best_score = -1;
    best_quad = None

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad = approx.reshape(4, 2).astype(np.float32)
            area = cv2.contourArea(quad);
            area_score = np.clip(area / (w * h), 0, 1)
            M = cv2.moments(quad);
            if M["m00"] == 0: continue
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            dist = np.linalg.norm(np.array([cx, cy]) - img_center)
            centrality_score = 1 - np.clip(dist / (max(w, h) / 2), 0, 1)
            rect = _order_points(quad);
            (tl, tr, br, bl) = rect
            width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
            height = (np.linalg.norm(tl - bl) + np.linalg.norm(tr - br)) / 2
            if min(width, height) < 1e-6: continue
            aspect_ratio = max(width, height) / min(width, height)
            aspect_score = math.exp(-0.5 * ((aspect_ratio - 1.4) ** 2))
            final_score = (area_score * 0.5) + (centrality_score * 0.4) + (aspect_score * 0.1)
            if final_score > best_score:
                best_score, best_quad = final_score, quad
    return best_quad


def stage3_texture_analysis(image: np.ndarray) -> Optional[np.ndarray]:
    # ... (Bu fonksiyon aynı)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gabor_kernels = [cv2.getGaborKernel(ksize=(31, 31), sigma=s, theta=t, lambd=10.0, gamma=0.5)
                     for s in (4.0, 6.0) for t in np.arange(0, np.pi, np.pi / 4)]
    accum = np.zeros_like(gray, dtype=np.float32)
    for kernel in gabor_kernels:
        filtered_img = cv2.filter2D(gray, cv2.CV_32F, kernel)
        np.maximum(accum, filtered_img, accum)
    accum = cv2.normalize(accum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, thresh = cv2.threshold(accum, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=5)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < image.shape[0] * image.shape[1] * 0.1: return None
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    return box.astype(np.float32)


def stage4_hough_clustered(image: np.ndarray) -> Optional[np.ndarray]:
    # ... (Bu fonksiyon aynı)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, int(min(image.shape[:2]) / 5))
    if lines is None: return None
    clusters = defaultdict(list)
    for line in lines:
        rho, theta = line[0]
        key = (round(theta * 10 / np.pi), round(rho / 60))
        clusters[key].append((rho, theta))
    clusters = [v for k, v in clusters.items() if len(v) > 3]
    clusters.sort(key=len, reverse=True)
    if len(clusters) < 4: return None
    avg_lines = [np.mean(cluster, axis=0) for cluster in clusters]
    h_lines, v_lines = [], []
    for line in avg_lines:
        _, theta = line
        if theta < np.pi / 4 or theta > 3 * np.pi / 4:
            v_lines.append(line)
        else:
            h_lines.append(line)
    if len(h_lines) < 2 or len(v_lines) < 2: return None
    h_lines.sort(key=lambda x: x[0]);
    v_lines.sort(key=lambda x: x[0])
    corners = [_line_intersection(h_lines[0], v_lines[0]), _line_intersection(h_lines[0], v_lines[-1]),
               _line_intersection(h_lines[-1], v_lines[-1]), _line_intersection(h_lines[-1], v_lines[0])]
    if any(c is None for c in corners): return None
    quad = np.array(corners, dtype=np.float32)
    if cv2.contourArea(quad) < image.shape[0] * image.shape[1] * 0.1: return None
    return quad


# -----------------------------------------------------------------------------
# 3. Ana Bileşen (Değişiklik Yok)
# -----------------------------------------------------------------------------
class PerspectiveCorrection(Component):
    # ... (İçi tamamen aynı)
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
            "Hızlı Gözcü": stage1_simple_contour,
            "Akıllı Yargıç": stage2_scored_contour,
            "Doku Avcısı": stage3_texture_analysis,
            "Çizgi Dedektifi": stage4_hough_clustered
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
# 4. Çalıştırıcı
# -----------------------------------------------------------------------------
Executor(sys.argv[1]).run()