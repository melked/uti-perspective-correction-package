import os
import sys
import cv2
import numpy as np
from typing import List, Callable

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveTransformation.src.utils.response import build_response
from components.PerspectiveTransformation.src.models.PackageModel import PackageModel


# -------------------- Helper Functions --------------------

def order_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(round(max(widthA, widthB)))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(round(max(heightA, heightB)))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LANCZOS4)


def full_image_quad(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)


def find_quad_from_contours(binary_img: np.ndarray, ref_image: np.ndarray, min_area_ratio=0.05) -> np.ndarray:
    contours, _ = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return full_image_quad(ref_image)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    img_area = ref_image.shape[0] * ref_image.shape[1]
    min_area = img_area * min_area_ratio

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype(np.float32)
    return full_image_quad(ref_image)


def unsharp_mask(image: np.ndarray, ksize=(5, 5), strength=1.5) -> np.ndarray:
    blur = cv2.GaussianBlur(image, ksize, 0)
    return cv2.addWeighted(image, 1 + strength, blur, -strength, 0)


def gamma_correction(image: np.ndarray, gamma=1.5) -> np.ndarray:
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)


def adaptive_contrast(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clip_limit = 2.0 if np.mean(L) < 100 else 3.0
    L = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8)).apply(L)
    img = cv2.cvtColor(cv2.merge((L, A, B)), cv2.COLOR_LAB2BGR)
    mean_gray = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    if mean_gray < 80: gamma = 1.8
    elif mean_gray > 180: gamma = 0.6
    else: gamma = 1.0
    return gamma_correction(img, gamma)


def preprocess_for_edges(image: np.ndarray) -> np.ndarray:
    img = cv2.bilateralFilter(image, 9, 75, 75)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    L = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(L)
    img = cv2.cvtColor(cv2.merge((L, A, B)), cv2.COLOR_LAB2BGR)
    return unsharp_mask(img)


# -------------------- Detection Pipelines --------------------

def detect_lab_range(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    mask = cv2.inRange(A, 130, 170) | cv2.inRange(B, 120, 160)
    L_blur = cv2.GaussianBlur(L, (5, 5), 0)
    light_mask = cv2.adaptiveThreshold(L_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 15, 5)
    edges = cv2.Canny(L_blur, 40, 120)
    combined = cv2.bitwise_and(cv2.bitwise_or(light_mask, edges), cv2.bitwise_not(mask))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    return combined


def detect_clahe_canny(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe_img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    edges = cv2.Canny(clahe_img, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)


def detect_bright_blur(image: np.ndarray) -> np.ndarray:
    img = gamma_correction(image, 1.8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe_img = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    sharp = unsharp_mask(clahe_img)
    edges = cv2.Canny(sharp, 30, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)


def detect_hough(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength=50, maxLineGap=10)
    if lines is None or len(lines) < 4:
        return full_image_quad(image)
    points = np.vstack([lines[:, 0, :2], lines[:, 0, 2:]])
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    return np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.float32)


# -------------------- Candidate Detection --------------------

def detect_document_candidates(image: np.ndarray) -> List[np.ndarray]:
    img = adaptive_contrast(image)
    img = preprocess_for_edges(img)

    pipelines: List[Callable[[np.ndarray], np.ndarray]] = [
        lambda img: find_quad_from_contours(detect_lab_range(img), img),
        lambda img: find_quad_from_contours(detect_clahe_canny(img), img),
        lambda img: find_quad_from_contours(detect_bright_blur(img), img),
        detect_hough
        # Buraya diğer pipeline’ları ekleyebilirsin: mask_background_complex, dark_object, gradient_magnitude vb.
    ]

    candidates = []
    for detector in pipelines:
        try:
            quad = detector(img)
            if not np.allclose(quad, full_image_quad(image), atol=1):
                candidates.append(quad)
        except Exception as e:
            print(f"Detector error: {e}")

    if not candidates:
        candidates.append(full_image_quad(image))
    return candidates


def score_quad(image: np.ndarray, quad: np.ndarray) -> float:
    # Basit scoring: alan oranı + kenar yoğunluğu
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)
    edge_density = np.sum(cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 50, 150) & mask) / 255
    area_score = cv2.contourArea(quad.astype(np.int32))
    return edge_density * 0.7 + area_score * 0.3


def select_best_quad(image: np.ndarray, candidates: List[np.ndarray]) -> np.ndarray:
    best_quad = full_image_quad(image)
    best_score = -1
    for quad in candidates:
        s = score_quad(image, quad)
        if s > best_score:
            best_score = s
            best_quad = quad
    return best_quad


# -------------------- Component --------------------

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
        warped = four_point_transform(src_img, best_quad)

        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)

        self.context["src_quad"] = best_quad.tolist()
        self.context["output_size"] = [warped.shape[1], warped.shape[0]]

        return build_response(context=self)


Executor(sys.argv[1]).run()
