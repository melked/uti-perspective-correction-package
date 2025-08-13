import os
import sys
import cv2
import numpy as np
from typing import List, Dict
from scipy.spatial import distance

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveCorrection.src.utils.response import build_response
from components.PerspectiveCorrection.src.models.PackageModel import PackageModel

# ----------------------------
# Temel Fonksiyonlar
# ----------------------------
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
    dst = np.array([[0, 0], [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LANCZOS4)
    return warped

def unsharp_mask(image: np.ndarray, ksize=(5, 5), strength=1.5) -> np.ndarray:
    blur = cv2.GaussianBlur(image, ksize, 0)
    return cv2.addWeighted(image, 1 + strength, blur, -strength, 0)

def gamma_correction(image: np.ndarray, gamma=1.5) -> np.ndarray:
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

def adaptive_contrast_enhancement(image: np.ndarray, clip_limit=None) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    if clip_limit is None:
        clip_limit = 2.0 if np.mean(L) < 100 else 3.0
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(L)
    lab_clahe = cv2.merge((cl, A, B))
    img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    mean_gray = np.mean(cv2.cvtColor(img_clahe, cv2.COLOR_BGR2GRAY))
    gamma_val = 1.0
    if mean_gray < 80:
        gamma_val = 1.8
    elif mean_gray > 180:
        gamma_val = 0.6
    return gamma_correction(img_clahe, gamma_val)

def preprocess_variants(img: np.ndarray, params: Dict = None) -> List[Dict]:
    variants = []
    if params is None: params = {}
    canny_thresholds = params.get("canny", [(50, 150)])
    blur_kernels = params.get("blur_kernels", [(5, 5)])
    unsharp_strength = params.get("unsharp_strength", 1.5)
    gamma_values = params.get("gamma_values", [0.6, 0.8, 1.2])

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Normal CLAHE + Canny
    clahe_img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    for low, high in canny_thresholds:
        edges = cv2.Canny(clahe_img, low, high)
        variants.append({"name": f"normal_{low}_{high}", "edges": edges})

    # Agresif CLAHE + Unsharp + Canny
    unsharp_img = unsharp_mask(clahe_img, ksize=(5, 5), strength=unsharp_strength)
    for low, high in canny_thresholds:
        edges = cv2.Canny(unsharp_img, low, high)
        variants.append({"name": f"agresif_{low}_{high}", "edges": edges})

    # Yumuşak Blur + Adaptive Threshold
    for k in blur_kernels:
        blur_img = cv2.GaussianBlur(gray, k, 0)
        thr = cv2.adaptiveThreshold(blur_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
        variants.append({"name": f"yumusak_{k[0]}", "edges": thr})

    # Gamma + CLAHE + Canny
    for g in gamma_values:
        gamma_img = np.array(np.power(clahe_img / 255.0, g) * 255, dtype=np.uint8)
        for low, high in canny_thresholds:
            edges = cv2.Canny(gamma_img, low, high)
            variants.append({"name": f"gamma_{g}_{low}_{high}", "edges": edges})

    # Bilateral + Canny
    for d in [7, 9]:
        bilat_img = cv2.bilateralFilter(img, d, 75, 75)
        edges = cv2.Canny(bilat_img, 50, 150)
        variants.append({"name": f"bilateral_{d}", "edges": edges})

    return variants

def find_corners_from_edges(edges: np.ndarray) -> np.ndarray:
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength=30, maxLineGap=10)
    if lines is None: return None
    points = []
    for i, l1 in enumerate(lines):
        for j, l2 in enumerate(lines):
            if i >= j: continue
            xdiff = np.array([l1[0][0] - l1[0][2], l2[0][0] - l2[0][2]])
            ydiff = np.array([l1[0][1] - l1[0][3], l2[0][1] - l2[0][3]])

            def det(a, b):
                a = np.array(a, dtype=np.float64)
                b = np.array(b, dtype=np.float64)
                return a[0] * b[1] - a[1] * b[0]

            div = det(xdiff, ydiff)
            if div == 0: continue
            d = (det([l1[0][0], l1[0][1]], [l1[0][2], l1[0][3]]),
                 det([l2[0][0], l2[0][1]], [l2[0][2], l2[0][3]]))
            x = det(d, xdiff) / div
            y = det(d, ydiff) / div
            if 0 <= x < edges.shape[1] and 0 <= y < edges.shape[0]:
                points.append([x, y])
    if len(points) < 4: return None
    points = np.array(points, dtype=np.float32)
    # cornerSubPix güvenliği: img yerine edges değil, gray kullanılacak
    gray = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) if edges.ndim == 2 else edges
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    cv2.cornerSubPix(gray, points, (5, 5), (-1, -1), criteria)
    dists = distance.cdist(points, points)
    sumd = dists.sum(axis=1)
    idxs = np.argsort(sumd)[-4:]
    return points[idxs]

def score_quad(image: np.ndarray, corners: np.ndarray) -> float:
    h, w = image.shape[:2]
    cx, cy = np.mean(corners, axis=0)
    area = cv2.contourArea(corners.astype(np.float32))
    ratio = min(w, h) / max(w, h)
    center_score = 1 - (abs(cx - w / 2) / w + abs(cy - h / 2) / h) / 2
    return area * ratio * center_score

def select_best_quad(image: np.ndarray, candidate_quads: List[np.ndarray]) -> np.ndarray:
    best_quad = np.array([[0, 0], [image.shape[1] - 1, 0],
                          [image.shape[1] - 1, image.shape[0] - 1],
                          [0, image.shape[0] - 1]], dtype=np.float32)
    best_score = -1
    for quad in candidate_quads:
        score = score_quad(image, quad)
        if score > best_score:
            best_score = score
            best_quad = quad
    return best_quad

def correct_perspective_optimized(img: np.ndarray, params: Dict = None) -> np.ndarray:
    priority_pipelines = ['normal', 'agresif']
    secondary_pipelines = ['yumusak', 'bilateral', 'gamma']

    variants = preprocess_variants(img, params)
    candidate_quads = []

    for name in priority_pipelines:
        for v in variants:
            if v['name'].startswith(name):
                corners = find_corners_from_edges(v["edges"])
                if corners is not None:
                    candidate_quads.append(corners)
        if candidate_quads: break

    if not candidate_quads:
        for name in secondary_pipelines:
            for v in variants:
                if v['name'].startswith(name):
                    corners = find_corners_from_edges(v["edges"])
                    if corners is not None:
                        candidate_quads.append(corners)
            if candidate_quads: break

    best_quad = select_best_quad(img, candidate_quads)
    warped = four_point_transform(img, best_quad)
    return warped

# ----------------------------
# Component
# ----------------------------
class PerspectiveCorrection(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")
        self.params = self.request.get_param("params", None)

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
        src_img = self._prepare_image(img_obj.value)
        result_img = correct_perspective_optimized(src_img, self.params)
        img_obj.value = result_img
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)
        return build_response(context=self)

if __name__ == "__main__":
    Executor(sys.argv[1]).run()
