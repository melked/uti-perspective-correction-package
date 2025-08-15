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


# ---------------- Utility Functions ---------------- #

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


def _unsharp_mask(image, ksize=(5, 5), strength=1.5):
    blur = cv2.GaussianBlur(image, ksize, 0)
    return cv2.addWeighted(image, 1 + strength, blur, -strength, 0)


def _gamma_correction(image: np.ndarray, gamma=1.5) -> np.ndarray:
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)


# ---------------- Preprocessing ---------------- #

def _adaptive_contrast_enhancement(image: np.ndarray, clip_limit=3.0) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(L)
    return cv2.cvtColor(cv2.merge((cl, A, B)), cv2.COLOR_LAB2BGR)


def _preprocess_image_for_edges(image: np.ndarray) -> np.ndarray:
    bilateral = cv2.bilateralFilter(image, 9, 75, 75)
    lab = cv2.cvtColor(bilateral, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(L)
    img_clahe = cv2.cvtColor(cv2.merge((cl, A, B)), cv2.COLOR_LAB2BGR)
    return _unsharp_mask(img_clahe)


# ---------------- Multi-Pipeline Candidate Detection ---------------- #

def _detect_candidates_with_varied_filters(image: np.ndarray) -> List[np.ndarray]:
    candidates = []
    clip_limits = [2.0, 3.0]
    gammas = [0.5, 1.0, 2.0]
    unsharp_strengths = [0.5, 1.0, 1.5]

    for clip in clip_limits:
        img_clahe = _adaptive_contrast_enhancement(image, clip_limit=clip)
        img_pre = _preprocess_image_for_edges(img_clahe)
        for gamma in gammas:
            img_gamma = _gamma_correction(img_pre, gamma)
            for strength in unsharp_strengths:
                img_final = _unsharp_mask(img_gamma, strength=strength)
                try:
                    gray = cv2.cvtColor(img_final, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, 30, 120)
                    quad = _find_quad_from_contours(edges, image)
                    candidates.append(quad)
                except:
                    continue
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
    score = sum([1 - abs(a - 90) / 90 for a in angles]) / 4
    return score


def _convexity_score(quad: np.ndarray) -> float:
    return 1.0 if cv2.isContourConvex(quad.astype(np.float32)) else 0.0


def _area_score(quad: np.ndarray, image: np.ndarray) -> float:
    area = cv2.contourArea(quad.astype(np.float32))
    img_area = image.shape[0] * image.shape[1]
    return min(area / img_area, 1.0)


def _shape_score(quad: np.ndarray) -> float:
    d1 = np.linalg.norm(quad[0] - quad[1])
    d2 = np.linalg.norm(quad[1] - quad[2])
    d3 = np.linalg.norm(quad[2] - quad[3])
    d4 = np.linalg.norm(quad[3] - quad[0])
    lengths = np.array([d1, d2, d3, d4])
    return 1.0 - np.std(lengths) / np.mean(lengths) if np.mean(lengths) > 0 else 0.0


def _score_quad(quad: np.ndarray, image: np.ndarray) -> float:
    area_s = _area_score(quad, image)
    conv_s = _convexity_score(quad)
    angle_s = _angle_score(quad)
    shape_s = _shape_score(quad)
    total_score = 0.4 * area_s + 0.2 * conv_s + 0.2 * angle_s + 0.2 * shape_s
    return total_score


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
        candidates = _detect_candidates_with_varied_filters(src_img)
        best_quad = select_best_quad(src_img, candidates)
        warped = _four_point_transform(src_img, best_quad)
        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)
        self.context["src_quad"] = best_quad.tolist()
        self.context["output_size"] = [warped.shape[1], warped.shape[0]]
        return build_response(context=self)


Executor(sys.argv[1]).run()
