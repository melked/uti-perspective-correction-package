import os
import sys
import cv2
import numpy as np
from typing import List

from PIL import Image as PILImage



sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveCorrection.src.utils.response import build_response
from components.PerspectiveCorrection.src.models.PackageModel import PackageModel


class Params:
    def __init__(self,
                 clahe_clip=2.0,
                 clahe_grid=(8, 8),
                 morph_kernel=5,
                 min_area_ratio=0.02,
                 canny_sigma=0.33):
        self.clahe_clip = clahe_clip
        self.clahe_grid = clahe_grid
        self.morph_kernel = morph_kernel
        self.min_area_ratio = min_area_ratio
        self.canny_sigma = canny_sigma


def _order_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1); diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]
    return rect


def _four_point_transform(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect
    wA = np.linalg.norm(br - bl); wB = np.linalg.norm(tr - tl)
    hA = np.linalg.norm(tr - br); hB = np.linalg.norm(tl - bl)
    maxW = int(round(max(wA, wB))); maxH = int(round(max(hA, hB)))
    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (maxW, maxH), flags=cv2.INTER_LANCZOS4)


def _full_image_quad(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)


def _adaptive_canny(gray: np.ndarray, sigma=0.33):
    v = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(gray, lower, upper)


def _preprocess_for_edges(img: np.ndarray, params: Params) -> np.ndarray:
    # CLAHE on L channel (LAB) + mild denoise + unsharp
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=params.clahe_clip, tileGridSize=params.clahe_grid)
    L2 = clahe.apply(L)
    lab2 = cv2.merge((L2, A, B))
    img2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    den = cv2.bilateralFilter(img2, 9, 75, 75)
    blur = cv2.GaussianBlur(den, (0, 0), 3)
    sharp = cv2.addWeighted(den, 1.5, blur, -0.5, 0)
    return sharp


def _find_quad_from_contours(binary: np.ndarray, ref_img: np.ndarray, min_area_ratio=0.02) -> np.ndarray:
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _full_image_quad(ref_img)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    img_area = ref_img.shape[0] * ref_img.shape[1]
    min_area = img_area * min_area_ratio
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype(np.float32)
    return _full_image_quad(ref_img)


# --- a few compact variants that cover most cases ---
def _variant_clahe_canny(img: np.ndarray, params: Params) -> np.ndarray:
    pre = _preprocess_for_edges(img, params)
    gray = cv2.cvtColor(pre, cv2.COLOR_BGR2GRAY)
    edges = _adaptive_canny(gray, sigma=params.canny_sigma)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (params.morph_kernel, params.morph_kernel))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    return _find_quad_from_contours(closed, img, min_area_ratio=params.min_area_ratio)


def _variant_inverse_thresh(img: np.ndarray, params: Params) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    return _find_quad_from_contours(morph, img, min_area_ratio=params.min_area_ratio)


def _variant_hough_box(img: np.ndarray, params: Params) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = _adaptive_canny(gray, sigma=params.canny_sigma)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=20)
    if lines is None:
        return _full_image_quad(img)
    pts = np.vstack([lines[:, 0, :2], lines[:, 0, 2:]])
    x0, y0 = pts.min(axis=0); x1, y1 = pts.max(axis=0)
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def _variant_dark_object(img: np.ndarray, params: Params) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB); L, _, _ = cv2.split(lab)
    _, mask = cv2.threshold(L, 90, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return _find_quad_from_contours(closed, img, min_area_ratio=params.min_area_ratio)


def detect_document_candidates(img: np.ndarray, params: Params) -> List[np.ndarray]:
    variants = [
        _variant_clahe_canny,
        _variant_inverse_thresh,
        _variant_hough_box,
        _variant_dark_object
    ]
    candidates = []
    for v in variants:
        try:
            quad = v(img, params)
            if not np.allclose(quad, _full_image_quad(img), atol=1):
                candidates.append(quad)
        except Exception:
            continue
    if not candidates:
        candidates.append(_full_image_quad(img))
    return candidates


def _score_quad(img: np.ndarray, quad: np.ndarray) -> float:
    # quick score: combination of area ratio and edge coverage inside quad
    h, w = img.shape[:2]
    img_area = h * w
    rect = _order_points(quad)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [rect.astype(np.int32)], 255)
    area = cv2.contourArea(rect.astype(np.int32))
    area_score = area / (img_area + 1e-9)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_inside = (cv2.countNonZero(cv2.bitwise_and(edges, edges, mask=mask)) / (area + 1e-9))
    return area_score * 0.7 + min(edge_inside, 0.01) * 30.0  # tuned weights


def select_best_quad(img: np.ndarray, candidates: List[np.ndarray]) -> np.ndarray:
    best = _full_image_quad(img); best_score = -1.0
    for q in candidates:
        s = _score_quad(img, q)
        if s > best_score:
            best_score = s; best = q
    return best


class PerspectiveTransformation(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.context = {}
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")
        params_data = self.request.get_param("params", None)
        self.params = Params(**params_data) if params_data else Params()

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def _prepare_image(self, img):
        if img is None or img.size == 0:
            raise ValueError("Input image is empty.")
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
            raise ValueError("No input image provided.")
        src = self._prepare_image(img_obj.value)

        candidates = detect_document_candidates(src, self.params)
        best_quad = select_best_quad(src, candidates)
        warped = _four_point_transform(src, best_quad)

        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)

        self.context["src_quad"] = best_quad.tolist()
        self.context["output_size"] = [warped.shape[1], warped.shape[0]]
        return build_response(context=self)


if __name__ == "__main__":
    Executor(sys.argv[1]).run()
