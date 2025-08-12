import os
import sys
import cv2
import numpy as np

# sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../')) # Removed this line

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveCorrection.src.utils.response import build_response
from components.PerspectiveCorrection.src.models.PackageModel import PackageModel


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def find_intersections(lines):
    """Finds intersection points from a set of lines."""
    intersections = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            line1 = lines[i][0]
            line2 = lines[j][0]

            rho1, theta1 = line1
            rho2, theta2 = line2

            A = np.array([
                [np.cos(theta1), np.sin(theta1)],
                [np.cos(theta2), np.sin(theta2)]
            ])
            b = np.array([[rho1], [rho2]])

            det = np.linalg.det(A)
            if det == 0:
                continue # Parallel lines

            x, y = np.linalg.solve(A, b)
            intersections.append((x[0], y[0]))
    return np.array(intersections)

def filter_and_select_corners(intersections, image_shape, quad_approx, tolerance=20):
    """Filters intersection points to find the four corners closest to the initial approximation."""
    h, w = image_shape[:2]
    corners = []
    for approx_point in quad_approx.reshape(4, 2):
        distances = np.linalg.norm(intersections - approx_point, axis=1)
        closest_intersection_index = np.argmin(distances)
        closest_intersection = intersections[closest_intersection_index]

        # Add a tolerance check to ensure the intersection is reasonably close
        if distances[closest_intersection_index] < tolerance:
             corners.append(closest_intersection)

    # If we didn't find 4 corners, return None
    if len(corners) != 4:
        return None

    return np.array(corners, dtype="float32")



# -----------------------------
# Mükemmel Perspektif Düzeltme (Kenar Kesişimi Tabanlı)
# -----------------------------
def correct_perspective_advanced(image: np.ndarray, params: dict = None) -> np.ndarray:
    """
    Kenar kesişimi tabanlı perspektif düzeltme.
    params (opsiyonel): {
      "canny_low_threshold": 50,
      "canny_high_threshold": 150,
      "hough_rho": 1,
      "hough_theta": np.pi / 180,
      "hough_threshold": 100,
      "intersection_tolerance": 20,
      "min_area_ratio": 0.01,
      "max_area_ratio": 0.95,
      "corner_subpix_win": (5,5),
      "corner_subpix_iter": 40
    }
    """
    # ---------- params ----------
    if params is None:
        params = {}
    CANNY_LOW = params.get("canny_low_threshold", 50)
    CANNY_HIGH = params.get("canny_high_threshold", 150)
    HOUGH_RHO = params.get("hough_rho", 1)
    HOUGH_THETA = params.get("hough_theta", np.pi / 180)
    HOUGH_THRESHOLD = params.get("hough_threshold", 100)
    INTERSECTION_TOLERANCE = params.get("intersection_tolerance", 20)
    MIN_AREA_RATIO = params.get("min_area_ratio", 0.01)
    MAX_AREA_RATIO = params.get("max_area_ratio", 0.95)
    SUBPIX_WIN = params.get("corner_subpix_win", (5, 5))
    SUBPIX_ITERS = params.get("corner_subpix_iter", 40)


    # ---------- Preprocessing ----------
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)

    # ---------- Find Lines (Hough Transform) ----------
    lines = cv2.HoughLines(edges, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD)

    if lines is None:
        # Fallback to original image if no lines are found
        return image

    # ---------- Find Intersections ----------
    intersections = find_intersections(lines)

    if intersections.size == 0:
         # Fallback to original image if no intersections are found
         return image

    # ---------- Initial Quad Approximation (using contours as a hint) ----------
    # This helps guide the intersection filtering
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quad_approx = None
    if contours:
        # Find the largest contour and approximate a polygon
        largest_contour = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)
        if len(approx) == 4:
            quad_approx = approx

    if quad_approx is None:
        # If contour approximation failed, use image corners as a very rough hint
        h, w = image.shape[:2]
        quad_approx = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32).reshape(4,1,2)


    # ---------- Filter and Select Corners from Intersections ----------
    corners = filter_and_select_corners(intersections, image.shape, quad_approx, INTERSECTION_TOLERANCE)

    if corners is None:
        # Fallback if filtering didn't yield 4 corners
        return image

    # ---------- Refine Corners (Subpixel) ----------
    corners = corners.reshape(-1, 1, 2).astype(np.float32)
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, SUBPIX_ITERS, 0.01)
    try:
        cv2.cornerSubPix(gray, corners, SUBPIX_WIN, (-1,-1), term)
    except Exception:
        # fallback: skip subpix if it fails
        pass
    corners = corners.reshape(4,2)

    # ---------- Order Points ----------
    rect = order_points(corners)

    # ---------- Validate Geometry (Optional but Recommended) ----------
    # You can add checks here for angle consistency, aspect ratio, etc.
    # For now, we rely on the intersection filtering and subpixel refinement.
    # You might add:
    # - Check if the area of the found quad is within reasonable bounds
    # - Check if angles are close to 90 degrees

    # Check area bounds on full-res
    full_h, full_w = gray.shape[:2]
    full_area = cv2.contourArea(rect.reshape(4,1,2))
    if full_area < (MIN_AREA_RATIO * full_w * full_h) or full_area > (MAX_AREA_RATIO * full_w * full_h):
        # Fallback if the found area is too small or too large
        return image

    # ---------- Warp Perspective ----------
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(round(max(widthA, widthB)))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(round(max(heightA, heightB)))

    if maxWidth < 10 or maxHeight < 10:
        return image # Prevent tiny outputs

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LANCZOS4)

    # Store the found corners in context for potential debugging/visualization
    try:
        # Convert corners to a serializable format if needed for context
        self.context["src_quad"] = rect.tolist()
    except Exception:
        pass


    return warped

class PerspectiveCorrection(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")
        self.params = self.request.get_param("params", None)
        self.context = {} # Initialize context here

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def _prepare_image(self, img):
        if img is None or img.size == 0:
            raise ValueError("Input image is empty or None.")
        if img.dtype != np.uint8:

            if img.max() <= 1.0 and np.issubdtype(img.dtype, np.floating):
                 img = (img * 255).astype(np.uint8)
            else:

                 img = img.astype(np.uint8)

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def run(self):
        img_obj = Image.get_frame(img=self.image, redis_db=self.redis_db)
        if img_obj is None or img_obj.value is None:
            raise ValueError("No input image provided or failed to load.")

        src = self._prepare_image(img_obj.value)

        warped = correct_perspective_advanced(src, self.params)

        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)

        try:
             self.context["output_size"] = [warped.shape[1], warped.shape[0]]
        except Exception:
             pass


        return build_response(context=self)


if __name__ == "__main__":
    Executor(sys.argv[1]).run()