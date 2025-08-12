import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveCorrection.src.utils.response import build_response
from components.PerspectiveCorrection.src.models.PackageModel import PackageModel


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def refine_corners_with_harris(gray, corners, search_radius=10):
    gray_f = np.float32(gray)
    dst = cv2.cornerHarris(gray_f, blockSize=5, ksize=3, k=0.04)
    dst = cv2.dilate(dst, None)

    refined = []
    for pt in corners.reshape(4, 2):
        x, y = int(pt[0]), int(pt[1])
        y_min, y_max = max(0, y - search_radius), min(dst.shape[0], y + search_radius)
        x_min, x_max = max(0, x - search_radius), min(dst.shape[1], x + search_radius)
        search_area = dst[y_min:y_max, x_min:x_max]

        if search_area.size == 0:
            refined.append([x, y])
            continue

        _, _, _, max_loc = cv2.minMaxLoc(search_area)
        refined_x = x_min + max_loc[0]
        refined_y = y_min + max_loc[1]
        refined.append([refined_x, refined_y])

    return np.array(refined, dtype=np.float32)


def validate_and_refine_quad(contour, edges, gray, img_shape):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.015 * peri, True)  # Daha hassas epsilon

    if len(approx) != 4:
        return None

    if not cv2.isContourConvex(approx):
        return None

    area = cv2.contourArea(approx)
    h, w = img_shape[:2]
    if area < 0.01 * w * h or area > 0.95 * w * h:
        return None

    pts = approx.reshape(4, 2)
    widths = [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]
    if min(widths) / max(widths) < 0.3:
        return None

    mask = np.zeros_like(edges)
    cv2.drawContours(mask, [approx], -1, 255, 2)
    overlap = cv2.countNonZero(cv2.bitwise_and(mask, edges))
    total_edge = cv2.countNonZero(mask)
    if total_edge == 0 or (overlap / total_edge) < 0.6:
        return None

    refined_corners = refine_corners_with_harris(gray, approx)
    refined_corners = order_points(refined_corners)

    return refined_corners.reshape((4, 1, 2))


def preprocess_variants(gray):
    variants = []

    # 1. CLAHE + Canny (Normal)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    edges = cv2.Canny(clahe_img, 50, 150)
    variants.append(("clahe_canny", edges))

    # 2. Unsharp Mask + CLAHE + Canny (Agresif)
    blur = cv2.GaussianBlur(clahe_img, (9, 9), 10)
    unsharp = cv2.addWeighted(clahe_img, 1.5, blur, -0.5, 0)
    edges_unsharp = cv2.Canny(unsharp, 50, 150)
    variants.append(("unsharp_canny", edges_unsharp))

    # 3. Gaussian Blur + Adaptive Threshold (Yumuşak)
    blur_soft = cv2.GaussianBlur(gray, (5, 5), 0)
    adaptive_thresh = cv2.adaptiveThreshold(blur_soft, 255,
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY,
                                            11, 2)
    variants.append(("adaptive_thresh", adaptive_thresh))

    return variants


def score_quad(quad, img_shape):
    pts = quad.reshape(4, 2)
    h, w = img_shape[:2]
    center = np.array([w/2, h/2])

    area = cv2.contourArea(quad)
    area_score = 1 - abs(area - 0.25*w*h) / (0.25*w*h)

    widths = [np.linalg.norm(pts[i] - pts[(i+1)%4]) for i in range(4)]
    height_diff = abs(widths[0] - widths[2]) / max(widths[0], widths[2])
    width_diff = abs(widths[1] - widths[3]) / max(widths[1], widths[3])
    rect_score = 1 - (height_diff + width_diff)/2

    cnt_center = np.mean(pts, axis=0)
    dist_center = np.linalg.norm(cnt_center - center)
    max_dist = np.linalg.norm(center)
    center_score = 1 - (dist_center / max_dist)

    return area_score + rect_score + center_score


def full_image_quad(image):
    h, w = image.shape[:2]
    return np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32).reshape((4, 1, 2))


def detect_best_quad(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    candidates = []

    variants = preprocess_variants(gray)

    for name, edges in variants:
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            quad = validate_and_refine_quad(cnt, edges, gray, image.shape)
            if quad is not None:
                score = score_quad(quad, image.shape)
                candidates.append((score, quad))

    if not candidates:
        # Fallback: En büyük kontur (approx 4 köşe)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            biggest = max(contours, key=cv2.contourArea)
            peri = cv2.arcLength(biggest, True)
            approx = cv2.approxPolyDP(biggest, 0.02 * peri, True)
            if len(approx) == 4:
                candidates.append((0, approx))

    if not candidates:
        return full_image_quad(image)

    best = max(candidates, key=lambda x: x[0])[1]
    return best


def four_point_transform(image, pts):
    rect = order_points(pts.reshape(4, 2))
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
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


class PerspectiveCorrection(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def _prepare_image(self, img):
        if img is None or img.size == 0:
            raise ValueError("Input image is empty or None.")
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
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

        src_img = self._prepare_image(img_obj.value)

        best_quad = detect_best_quad(src_img)
        warped = four_point_transform(src_img, best_quad)

        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)

        self.context = {
            "src_quad": best_quad.reshape(4, 2).tolist(),
            "output_size": [warped.shape[1], warped.shape[0]]
        }

        return build_response(context=self)


if __name__ == "__main__":
    Executor(sys.argv[1]).run()
