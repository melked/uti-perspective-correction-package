import os
import sys
import cv2
import numpy as np
from typing import List

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveCorrection.src.utils.response import build_response
from components.PerspectiveCorrection.src.models.PackageModel import PackageModel


# ---------------- Quad Helpers ---------------- #

def _order_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(round(max(widthA, widthB)))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(round(max(heightA, heightB)))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LANCZOS4)
    return warped


def _full_image_quad(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)


def _find_quad_from_contours(binary_img: np.ndarray, ref_image: np.ndarray, min_area_ratio=0.03,
                             max_area_ratio=0.95) -> np.ndarray:
    if binary_img is None or binary_img.size == 0:
        return _full_image_quad(ref_image)
    if len(binary_img.shape) == 3:
        binary_img = cv2.cvtColor(binary_img, cv2.COLOR_BGR2GRAY)
    if binary_img.dtype != np.uint8:
        binary_img = cv2.normalize(binary_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    contours, _ = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _full_image_quad(ref_image)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    img_area = ref_image.shape[0] * ref_image.shape[1]
    min_area = img_area * min_area_ratio
    max_area = img_area * max_area_ratio

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype(np.float32)

    return _full_image_quad(ref_image)


# ---------------- Preprocessing ---------------- #

def _unsharp_mask(image, ksize=(5, 5), strength=1.5):
    blur = cv2.GaussianBlur(image, ksize, 0)
    return cv2.addWeighted(image, 1 + strength, blur, -strength, 0)


def _gamma_correction(image: np.ndarray, gamma=1.5) -> np.ndarray:
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)


def _adaptive_contrast_enhancement(image: np.ndarray, clip_limit=3.0) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(L)
    return cv2.cvtColor(cv2.merge((cl, A, B)), cv2.COLOR_LAB2BGR)


def _preprocess_soft(image: np.ndarray) -> np.ndarray:   ### ADDED
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, 11, 2)


def _preprocess_medium(image: np.ndarray) -> np.ndarray:   ### ADDED
    img_clahe = _adaptive_contrast_enhancement(image, 2.0)
    gray = cv2.cvtColor(img_clahe, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, 50, 150)


def _preprocess_hard(image: np.ndarray) -> np.ndarray:   ### ADDED
    img_clahe = _adaptive_contrast_enhancement(image, 3.0)
    img_sharp = _unsharp_mask(img_clahe, strength=1.5)
    gray = cv2.cvtColor(img_sharp, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, 30, 120)


def _detect_candidates_with_varied_filters(image: np.ndarray) -> List[np.ndarray]:
    candidates = []

    # Run 3 pipelines (soft/medium/hard)
    for preprocess_fn in [_preprocess_soft, _preprocess_medium, _preprocess_hard]:
        try:
            binary = preprocess_fn(image)
            quad = _find_quad_from_contours(binary, image)
            candidates.append(quad)
        except:
            continue

    # Fallback: full image
    if not candidates:
        candidates.append(_full_image_quad(image))
    return candidates


# ---------------- Quad Scoring ---------------- #

def _angle_score(quad: np.ndarray) -> float:
    def angle(pt1, pt2, pt3):
        v1 = pt1 - pt2
        v2 = pt3 - pt2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        ang = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(ang)
    angles = [angle(quad[i], quad[(i + 1) % 4], quad[(i + 2) % 4]) for i in range(4)]
    return sum([1 - abs(a - 90) / 90 for a in angles]) / 4


def _convexity_score(quad: np.ndarray) -> float:
    return 1.0 if cv2.isContourConvex(quad.astype(np.float32)) else 0.0


def _area_score(quad: np.ndarray, image: np.ndarray) -> float:
    area = cv2.contourArea(quad.astype(np.float32))
    img_area = image.shape[0] * image.shape[1]
    return min(area / img_area, 1.0)


def _shape_score(quad: np.ndarray) -> float:
    d = [np.linalg.norm(quad[i] - quad[(i + 1) % 4]) for i in range(4)]
    return 1.0 - np.std(d) / np.mean(d) if np.mean(d) > 0 else 0.0


def _aspect_ratio_score(quad: np.ndarray) -> float:   ### ADDED
    w1 = np.linalg.norm(quad[0] - quad[1])
    w2 = np.linalg.norm(quad[2] - quad[3])
    h1 = np.linalg.norm(quad[1] - quad[2])
    h2 = np.linalg.norm(quad[3] - quad[0])
    w, h = (w1 + w2) / 2, (h1 + h2) / 2
    if h == 0: return 0.0
    ratio = w / h
    ideal = 1.414  # A4 oranı
    return max(0.0, 1 - abs(ratio - ideal) / ideal)


def _brightness_contrast_score(quad: np.ndarray, image: np.ndarray) -> float:  ### ADDED
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [quad.astype(np.int32)], 255)
    inside = cv2.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), mask=mask)[0]
    outside = cv2.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), mask=cv2.bitwise_not(mask))[0]
    return 1.0 if inside > outside else 0.0


def _score_quad(quad: np.ndarray, image: np.ndarray) -> float:
    return (0.25 * _area_score(quad, image) +
            0.15 * _convexity_score(quad) +
            0.2 * _angle_score(quad) +
            0.15 * _shape_score(quad) +
            0.15 * _aspect_ratio_score(quad) +   # ADDED
            0.1 * _brightness_contrast_score(quad, image))   # ADDED


def select_best_quad(image: np.ndarray, candidates: List[np.ndarray]) -> np.ndarray:
    scored = [(quad, _score_quad(quad, image)) for quad in candidates]
    best = max(scored, key=lambda x: x[1])[0]
    return best


# ---------------- Component ---------------- #

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
        if img is None or img.size == 0:
            raise ValueError("Input image empty or None")
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
            raise ValueError("No input image provided or failed to load")
        src_img = self._prepare_image(img_obj.value)

        # Çoklu pipeline’dan aday quad üret
        candidates = _detect_candidates_with_varied_filters(src_img)

        # En iyisini seç
        best_quad = select_best_quad(src_img, candidates)

        # Warp et
        warped = _four_point_transform(src_img, best_quad)

        # Warp sonrası kontrast/gamma iyileştirme (ADDED)
        warped = _adaptive_contrast_enhancement(warped, 2.0)
        warped = _gamma_correction(warped, 1.2)

        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)
        self.context["src_quad"] = best_quad.tolist()
        self.context["output_size"] = [warped.shape[1], warped.shape[0]]
        return build_response(context=self)


Executor(sys.argv[1]).run()
