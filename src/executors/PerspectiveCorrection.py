import os
import sys
import cv2
import numpy as np

from typing import Optional, List, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveTransformation.src.utils.response import build_response
from components.PerspectiveTransformation.src.models.PackageModel import PackageModel

# ---------------------- Geometri Yardımcıları ----------------------
def _order_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect.astype(np.float32)

def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(round(max(widthA, widthB)))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(round(max(heightA, heightB)))
    maxWidth = max(10, maxWidth)
    maxHeight = max(10, maxHeight)
    dst = np.array([[0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LANCZOS4)
    return warped

def _full_image_quad(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

# ---------------------- Temel Filtreler ----------------------
def _unsharp_mask(image, ksize=(5, 5), strength=1.5):
    blur = cv2.GaussianBlur(image, ksize, 0)
    return cv2.addWeighted(image, 1 + strength, blur, -strength, 0)

def _gamma_correction(image: np.ndarray, gamma=1.5) -> np.ndarray:
    invGamma = 1.0 / max(1e-6, gamma)
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

def _auto_gamma_correction(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = float(np.mean(gray))
    if mean < 80:
        gamma = 1.8
    elif mean > 180:
        gamma = 0.6
    else:
        gamma = 1.0
    return _gamma_correction(image, gamma)

# ---------------------- Maske/Ön-işleme ----------------------
def _mask_background_lab_range(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    mask_a = cv2.inRange(A, 130, 170)
    mask_b = cv2.inRange(B, 120, 160)
    color_mask = cv2.bitwise_or(mask_a, mask_b)
    L_blur = cv2.GaussianBlur(L, (5, 5), 0)
    light_mask = cv2.adaptiveThreshold(L_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 15, 5)
    edge_mask = cv2.Canny(L_blur, 40, 120)
    combined = cv2.bitwise_or(light_mask, edge_mask)
    combined = cv2.bitwise_and(combined, cv2.bitwise_not(color_mask))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    return combined

def _mask_background_complex(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    _, mask_a = cv2.threshold(A, 135, 255, cv2.THRESH_BINARY_INV)
    _, mask_b = cv2.threshold(B, 135, 255, cv2.THRESH_BINARY)
    color_mask = cv2.bitwise_and(mask_a, mask_b)
    L_blur = cv2.GaussianBlur(L, (5, 5), 0)
    light_mask = cv2.adaptiveThreshold(L_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 15, 5)
    edge_mask = cv2.Canny(L_blur, 40, 120)
    combined = cv2.bitwise_or(light_mask, edge_mask)
    combined = cv2.bitwise_and(combined, color_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    return combined

def _adaptive_contrast_enhancement(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    mean_lum = float(np.mean(L))
    clip_limit = 2.0 if mean_lum < 100 else 3.0
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(L)
    lab_clahe = cv2.merge((cl, A, B))
    img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    mean_gray = float(np.mean(cv2.cvtColor(img_clahe, cv2.COLOR_BGR2GRAY)))
    if mean_gray < 80:
        gamma = 1.8
    elif mean_gray > 180:
        gamma = 0.6
    else:
        gamma = 1.0
    return _gamma_correction(img_clahe, gamma=gamma)

def _preprocess_image_for_edges(image: np.ndarray) -> np.ndarray:
    bilateral = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    lab = cv2.cvtColor(bilateral, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(L)
    lab_clahe = cv2.merge((cl, A, B))
    img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    sharpened = _unsharp_mask(img_clahe)
    return sharpened

def _auto_canny(image: np.ndarray, sigma=0.33):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    v = float(np.median(gray))
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(gray, lower, upper)

# ---------------------- Merkez-Öncelikli Dörtgen Bulma ----------------------
def _quad_angle_score(quad: np.ndarray) -> float:
    q = _order_points(quad)
    angles = []
    for i in range(4):
        p1, p2, p3 = q[(i-1) % 4], q[i], q[(i+1) % 4]
        v1, v2 = p1 - p2, p3 - p2
        den = (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cosang = np.clip(np.dot(v1, v2) / den, -1.0, 1.0)
        ang = np.degrees(np.arccos(cosang))
        angles.append(ang)
    # 90°'ye yakınlık
    return float(np.mean([max(0.0, 1.0 - abs(a - 90.0) / 45.0) for a in angles]))

def _center_score(quad: np.ndarray, img_shape: Tuple[int, int]) -> float:
    h, w = img_shape[:2]
    img_center = np.array([w/2.0, h/2.0])
    M = cv2.moments(quad.astype(np.float32))
    if M["m00"] == 0:
        return 0.0
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    dist = np.linalg.norm(np.array([cx, cy]) - img_center) / max(w, h)
    return float(max(0.0, 1.0 - dist))  # merkeze yakınsa yüksek

def _aspect_score(quad: np.ndarray) -> float:
    q = _order_points(quad)
    w = (np.linalg.norm(q[0] - q[1]) + np.linalg.norm(q[2] - q[3])) / 2.0
    h = (np.linalg.norm(q[1] - q[2]) + np.linalg.norm(q[0] - q[3])) / 2.0
    if min(w, h) < 1e-3:
        return 0.0
    r = max(w, h) / min(w, h)
    # Aşırı uzun/ince dikdörtgenleri biraz cezalandır, 1-3 arası iyi kabul
    return float(max(0.0, 1.0 - max(0.0, r - 3.0) / 5.0))

def _score_quad(image: np.ndarray, quad: np.ndarray) -> float:
    h, w = image.shape[:2]
    img_area = float(h * w)
    area = float(cv2.contourArea(quad.astype(np.float32)))
    if area <= 1.0:
        return 0.0
    area_ratio = np.clip(area / img_area, 0.0, 1.0)
    # 0.05–0.95 aralığı en iyi
    if area_ratio < 0.03 or area_ratio > 0.98:
        area_term = 0.1
    else:
        area_term = 0.6 * area_ratio + 0.4
    angle_term = _quad_angle_score(quad)
    center_term = _center_score(quad, image.shape)
    aspect_term = _aspect_score(quad)
    # toplam skor
    score = (area_term * 0.35) + (angle_term * 0.25) + (center_term * 0.25) + (aspect_term * 0.15)
    return float(score)

def _find_quad_from_contours(binary_img: np.ndarray, ref_image: np.ndarray, min_area_ratio=0.03) -> np.ndarray:
    # Her ihtimale karşı 8-bit tek kanal
    if binary_img.ndim == 3:
        binary_img = cv2.cvtColor(binary_img, cv2.COLOR_BGR2GRAY)
    _, bin8 = cv2.threshold(binary_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(bin8, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _centered_fallback(ref_image)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    img_area = ref_image.shape[0] * ref_image.shape[1]
    min_area = img_area * min_area_ratio

    best = None
    best_score = -1.0

    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            score = _score_quad(ref_image, approx.reshape(4, 2))
            if score > best_score:
                best_score = score
                best = approx.reshape(4, 2).astype(np.float32)

    if best is None:
        return _centered_fallback(ref_image)
    return best

def _centered_fallback(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    # Orta alanda güvenli bir %80 dikdörtgen
    margin_x = int(0.10 * w)
    margin_y = int(0.10 * h)
    return np.array([
        [margin_x, margin_y],
        [w - 1 - margin_x, margin_y],
        [w - 1 - margin_x, h - 1 - margin_y],
        [margin_x, h - 1 - margin_y]
    ], dtype=np.float32)

# ---------------------- Çeşitli Aday Üreticileri ----------------------
def _auto_detect_document_corners_sharpen_adaptive(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    sharpened = _unsharp_mask(blur)
    thresh = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    return _find_quad_from_contours(morph, image)

def _auto_detect_document_corners_clahe_canny(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    edges = cv2.Canny(clahe_img, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return _find_quad_from_contours(morph, image)

def _auto_detect_document_corners_bright_blur(image: np.ndarray) -> np.ndarray:
    gamma_corrected = _gamma_correction(image, gamma=1.8)
    gray = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    sharp = _unsharp_mask(clahe_img, ksize=(5, 5), strength=1.5)
    edges = cv2.Canny(sharp, 30, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    return _find_quad_from_contours(closed, image)

def _filter_lines_by_angle(lines, angle_tol=10):
    if lines is None:
        return None
    filtered = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        if (abs(angle - 0) < angle_tol) or (abs(angle - 90) < angle_tol) or (abs(angle - 180) < angle_tol):
            filtered.append(line)
    return np.array(filtered) if filtered else None

def _auto_detect_document_corners_hough_improved(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=10)
    lines = _filter_lines_by_angle(lines, angle_tol=15)
    if lines is None or len(lines) < 4:
        return _centered_fallback(image)
    all_points = np.vstack([lines[:, 0, :2], lines[:, 0, 2:]])
    x_min, y_min = np.min(all_points, axis=0)
    x_max, y_max = np.max(all_points, axis=0)
    return np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.float32)

def _texture_mask_gabor(image: np.ndarray, ksize=31, sigma=4.0, theta=np.pi/4, lambd=10.0, gamma=0.5) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    g_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
    filtered = cv2.filter2D(gray, cv2.CV_8UC3, g_kernel)
    _, mask = cv2.threshold(filtered, 50, 255, cv2.THRESH_BINARY)
    return mask

def _auto_detect_document_corners_lab_adaptive(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_L = clahe.apply(L)
    thresh_L = cv2.adaptiveThreshold(clahe_L, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 15, 2)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    combined = cv2.bitwise_or(thresh_L, edges)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    return _find_quad_from_contours(morph, image)

def _auto_detect_document_corners_color_segmentation(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_light = np.array([0, 0, 180], dtype=np.uint8)
    upper_light = np.array([180, 30, 255], dtype=np.uint8)
    mask_light = cv2.inRange(hsv, lower_light, upper_light)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    morph = cv2.morphologyEx(mask_light, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    return _find_quad_from_contours(morph, image)

def _auto_detect_document_corners_inverse_threshold(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    return _find_quad_from_contours(morph, image)

def _auto_detect_document_corners_gradient_magnitude(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, _ = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)
    mag_u8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, thresh = cv2.threshold(mag_u8, 50, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    return _find_quad_from_contours(morph, image)

def _auto_detect_document_corners_dark_object(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, _, _ = cv2.split(lab)
    _, dark_mask = cv2.threshold(L, 80, 255, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(dark_mask, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    return _find_quad_from_contours(closed, image)

# ---- ÖZEL: Beyaz arka plan + beyaz belge + düşük ışık için güçlü varyant ----
def _auto_detect_document_corners_white_on_white(image: np.ndarray) -> np.ndarray:
    # 1) Aydınlatma düzeltme: büyük blur ile arka plan tahminini çıkar
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=25, sigmaY=25)
    flat = cv2.addWeighted(gray, 1.5, bg, -0.5, 0)  # düzleştirilmiş aydınlatma

    # 2) CLAHE ile yerel kontrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    flat = clahe.apply(flat)

    # 3) Unsharp + Auto-Canny
    sharp = _unsharp_mask(flat, ksize=(5, 5), strength=1.2)
    edges = cv2.Canny(sharp, 0, 0)  # dummy
    # auto-canny eşiklerini median ile belirle
    v = float(np.median(sharp))
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))
    edges = cv2.Canny(sharp, lower, upper)

    # 4) Morfoloji
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

    return _find_quad_from_contours(opened, image)

# ---------------------- Aday Üret ve Seç ----------------------
def detect_document_candidates(image: np.ndarray) -> List[np.ndarray]:
    candidates = []
    img_corrected = _adaptive_contrast_enhancement(image)
    img_preprocessed = _preprocess_image_for_edges(img_corrected)

    variants = {
        "white_on_white": _auto_detect_document_corners_white_on_white,              # yeni
        "lab_range": lambda img: _find_quad_from_contours(_mask_background_lab_range(img), img),
        "color_segmentation": _auto_detect_document_corners_color_segmentation,
        "lab_adaptive": _auto_detect_document_corners_lab_adaptive,
        "sharpen_adaptive": _auto_detect_document_corners_sharpen_adaptive,
        "bright_blur": _auto_detect_document_corners_bright_blur,
        "clahe_canny": _auto_detect_document_corners_clahe_canny,
        "mask_background_complex": lambda img: _find_quad_from_contours(_mask_background_complex(img), img),
        "hough_improved": _auto_detect_document_corners_hough_improved,
        "inverse_threshold": _auto_detect_document_corners_inverse_threshold,
        "gradient_magnitude": _auto_detect_document_corners_gradient_magnitude,
        "dark_object": _auto_detect_document_corners_dark_object
    }

    for name, func in variants.items():
        try:
            quad = func(img_preprocessed)
            if not np.allclose(quad, _full_image_quad(image), atol=1):
                candidates.append(quad)
        except Exception as e:
            print(f"Error in {name} variant: {e}")

    if not candidates:
        candidates.append(_centered_fallback(image))

    return candidates

def select_best_quad(image: np.ndarray, candidates: List[np.ndarray]) -> np.ndarray:
    best_quad = _centered_fallback(image)
    best_score = -1.0
    for quad in candidates:
        score = _score_quad(image, quad)
        if score > best_score:
            best_score = score
            best_quad = quad
    return best_quad

# ---------------------- Bileşen ----------------------
class PerspectiveTransformation(Component):
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

        candidates = detect_document_candidates(src_img)
        best_quad = select_best_quad(src_img, candidates)

        warped = _four_point_transform(src_img, best_quad)

        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)

        self.context["src_quad"] = best_quad.tolist()
        self.context["output_size"] = [warped.shape[1], warped.shape[0]]

        return build_response(context=self)

# ---------------------- Çalıştır ----------------------
Executor(sys.argv[1]).run()
