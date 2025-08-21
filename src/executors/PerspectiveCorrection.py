import os
import sys
import cv2
import numpy as np
from typing import Optional, List, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveCorrection.src.utils.response import build_response
from components.PerspectiveCorrection.src.models.PackageModel import PackageModel

# -----------------------------------------------------------------------------
# 1. PARAMETER MANAGEMENT CLASS (NEW)
# -----------------------------------------------------------------------------
class Params:
    """A centralized class to manage all algorithm parameters."""
    def __init__(self, config=None):
        config = config or {}
        # General
        self.resize_longest_edge = config.get("resize_longest_edge", 1000)
        self.unsharp_strength = config.get("unsharp_strength", 1.5)
        # Scoring
        self.score_min_area_ratio = config.get("score_min_area_ratio", 0.05)
        self.score_max_area_ratio = config.get("score_max_area_ratio", 0.98)
        self.approx_poly_epsilon_ratio = config.get("approx_poly_epsilon_ratio", 0.02)
        # Thresholds
        self.high_confidence_threshold = config.get("high_confidence_threshold", 0.5)
        self.min_confidence_threshold = config.get("min_confidence_threshold", 0.2)
        # Boundary Strategy
        self.boundary_blur_ratio = config.get("boundary_blur_ratio", 5)

# -----------------------------------------------------------------------------
# 2. HELPER & GEOMETRY FUNCTIONS
# -----------------------------------------------------------------------------
def _order_points(pts: np.ndarray) -> np.ndarray:
    """Orders points: top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1); rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1); rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]
    return rect

def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> Optional[np.ndarray]:
    """Applies a four-point perspective transform to the image."""
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl); widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br); heightB = np.linalg.norm(tl - bl)
    maxWidth = max(int(widthA), int(widthB)); maxHeight = max(int(heightA), int(heightB))

    if maxWidth <= 10 or maxHeight <= 10: return None

    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LANCZOS4)

def _unsharp_mask(image: np.ndarray, strength: float) -> np.ndarray:
    """Sharpens the image using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, (0,0), 3)
    return cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)

# -----------------------------------------------------------------------------
# 3. EXPERT STRATEGIES (Candidate Generators)
# -----------------------------------------------------------------------------
def get_candidates_from_canny(image: np.ndarray) -> List[np.ndarray]:
    """Generates contour candidates using auto Canny edge detection."""
    gray = cv2.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    v = np.median(blurred); sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v)); upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(blurred, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def get_candidates_from_boundary(image: np.ndarray, params: Params) -> List[np.ndarray]:
    """Generates candidates by flattening the background illumination."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel_size = int(min(image.shape[:2]) / params.boundary_blur_ratio)
    if kernel_size % 2 == 0: kernel_size += 1
    blurred_bg = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    flattened = cv2.divide(gray, blurred_bg, scale=255)
    _, thresh = cv2.threshold(flattened, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# -----------------------------------------------------------------------------
# 4. CANDIDATE SCORING (REFINED)
# -----------------------------------------------------------------------------
def _score_candidate(contour: np.ndarray, params: Params, image_shape: tuple) -> float:
    """Scores a single contour based on geometric properties."""
    h, w = image_shape[:2]; total_area = w * h
    area = cv2.contourArea(contour)

    # Rule 1: Area must be within a reasonable range
    if not (params.score_min_area_ratio < area / total_area < params.score_max_area_ratio):
        return 0.0

    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, params.approx_poly_epsilon_ratio * peri, True)

    # Rule 2: Must be a convex quadrilateral
    if len(approx) != 4 or not cv2.isContourConvex(approx):
        return 0.0

    # Feature 1: Solidity (how non-concave the shape is)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity = area / hull_area if hull_area > 0 else 0

    # Feature 2: Centrality (how close it is to the image center)
    M = cv2.moments(approx)
    if M["m00"] == 0: return 0.0
    cx = M["m10"] / M["m00"]; cy = M["m01"] / M["m00"]
    centrality = 1.0 - (np.linalg.norm(np.array([cx, cy]) - np.array([w/2, h/2])) / (max(w, h)/2))

    # Feature 3: Aspect Ratio (should resemble paper)
    rect = _order_points(approx.reshape(4, 2))
    (tl, tr, br, bl) = rect
    width = (np.linalg.norm(tr-tl) + np.linalg.norm(br-bl)) / 2
    height = (np.linalg.norm(tl-bl) + np.linalg.norm(tr-br)) / 2
    if min(width, height) < 1: return 0.0
    aspect_ratio = max(width, height) / min(width, height)
    aspect_score = 1.0 if 1.1 < aspect_ratio < 2.2 else 0.5 # A4 is ~1.41, Letter is ~1.29

    # Weighted final score
    return (area/total_area * 0.4) + (centrality * 0.3) + (solidity * 0.2) + (aspect_score * 0.1)

# -----------------------------------------------------------------------------
# 5. MAIN COMPONENT (IMPROVED)
# -----------------------------------------------------------------------------
class PerspectiveCorrection(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.context = {}
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")
        params_data = self.request.get_param("params", {})
        self.params = Params(params_data)

    @staticmethod
    def bootstrap(config: dict) -> dict: return {}

    def _prepare_image(self, img: np.ndarray) -> np.ndarray:
        if img is None or img.size == 0: raise ValueError("Input image is empty or None.")
        if img.dtype != np.uint8: img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if img.ndim == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[-1] == 4: img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def run(self):
        img_obj = Image.get_frame(img=self.image, redis_db=self.redis_db)
        if img_obj is None or img_obj.value is None: raise ValueError("No input image provided or failed to load.")

        src_img_orig = self._prepare_image(img_obj.value)
        h, w = src_img_orig.shape[:2]

        # --- Pre-processing ---
        scale = self.params.resize_longest_edge / max(h, w) if max(h,w) > self.params.resize_longest_edge else 1
        work_img = cv2.resize(src_img_orig, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        work_img = _unsharp_mask(work_img, self.params.unsharp_strength)

        document_quad = None

        # --- Stage 1: Fast Scan ---
        print("Stage 1 (Fast Scan): Gathering candidates...")
        fast_candidates = []
        fast_candidates.extend(get_candidates_from_canny(work_img))
        fast_candidates.extend(get_candidates_from_boundary(work_img, self.params))

        best_candidate, best_score = None, self.params.high_confidence_threshold

        if fast_candidates:
            print(f"{len(fast_candidates)} candidates found. Searching for a high-confidence result...")
            scored_candidates = [(_score_candidate(c, self.params, work_img.shape), c) for c in fast_candidates]
            scored_candidates.sort(key=lambda x: x[0], reverse=True)

            if scored_candidates[0][0] > best_score:
                best_score, best_candidate = scored_candidates[0]
                print(f"High-confidence candidate found with score {best_score:.2f}. Finalizing.")
                peri = cv2.arcLength(best_candidate, True)
                document_quad = cv2.approxPolyDP(best_candidate, self.params.approx_poly_epsilon_ratio * peri, True)
                document_quad = document_quad.reshape(4, 2).astype(np.float32)

        # --- Stage 2: Deep Analysis (Fallback) ---
        if document_quad is None:
            print("No high-confidence candidate found. Analyzing all candidates...")
            if fast_candidates:
                # Use the best candidate found, even if it's below the high-confidence threshold
                best_deep_score, best_deep_candidate = scored_candidates[0]
                if best_deep_score > self.params.min_confidence_threshold:
                    print(f"Best available candidate found with score {best_deep_score:.2f}.")
                    peri = cv2.arcLength(best_deep_candidate, True)
                    document_quad = cv2.approxPolyDP(best_deep_candidate, self.params.approx_poly_epsilon_ratio * peri, True)
                    document_quad = document_quad.reshape(4, 2).astype(np.float32)

        # --- Final Transformation ---
        if document_quad is not None:
            document_quad /= scale # Scale quad back to original image size
            warped = _four_point_transform(src_img_orig, document_quad)
        else:
            warped = None

        if warped is None:
            print("All analyses failed. Using the original image as fallback.")
            warped = src_img_orig
            document_quad = np.array([[0,0], [w-1,0], [w-1,h-1], [0,h-1]], dtype=np.float32)

        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)
        self.context["src_quad"] = document_quad.tolist()
        self.context["output_size"] = [warped.shape[1], warped.shape[0]]
        return build_response(context=self)

# -----------------------------------------------------------------------------
# 6. EXECUTOR
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    Executor(sys.argv[1]).run()