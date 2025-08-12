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


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def validate_quad(contour, edges, img_shape):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
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
    if min(widths) / max(widths) < 0.2:
        return None

    mask = np.zeros_like(edges)
    cv2.drawContours(mask, [approx], -1, 255, 2)
    overlap = cv2.countNonZero(cv2.bitwise_and(mask, edges))
    total_edge = cv2.countNonZero(mask)
    if total_edge == 0 or (overlap / total_edge) < 0.6:
        return None

    return approx


def score_quad(quad, img_shape):
    pts = quad.reshape(4, 2)
    h, w = img_shape[:2]
    center = np.array([w / 2, h / 2])

    area = cv2.contourArea(quad)
    area_score = 1 - abs(area - 0.25 * w * h) / (0.25 * w * h)

    widths = [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]
    height_diff = abs(widths[0] - widths[2]) / max(widths[0], widths[2])
    width_diff = abs(widths[1] - widths[3]) / max(widths[1], widths[3])
    rect_score = 1 - (height_diff + width_diff) / 2

    cnt_center = np.mean(pts, axis=0)
    dist_center = np.linalg.norm(cnt_center - center)
    max_dist = np.linalg.norm(np.array([w / 2, h / 2]))
    center_score = 1 - (dist_center / max_dist)

    return area_score + rect_score + center_score


def preprocess_variants(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    variants = []

    # 1. CLAHE + Canny
    v1 = clahe.apply(gray)
    edges1 = cv2.Canny(v1, 50, 150)
    variants.append(("clahe_canny", edges1))

    # 2. Gaussian Blur + Adaptive Threshold
    blur2 = cv2.GaussianBlur(gray, (5, 5), 0)
    thr2 = cv2.adaptiveThreshold(blur2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)
    variants.append(("blur_adaptive_thresh", thr2))

    # 3. Unsharp Mask + Canny
    blur3 = cv2.GaussianBlur(gray, (9, 9), 10.0)
    unsharp3 = cv2.addWeighted(gray, 1.5, blur3, -0.5, 0)
    edges3 = cv2.Canny(unsharp3, 50, 150)
    variants.append(("unsharp_canny", edges3))

    # 4. Otsu Threshold
    _, thr4 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("otsu_thresh", thr4))

    # 5. Bilateral Filter + CLAHE + Canny
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe_bilateral = clahe.apply(bilateral)
    edges5 = cv2.Canny(clahe_bilateral, 50, 150)
    variants.append(("bilateral_clahe_canny", edges5))

    # 6. Inverse Threshold
    _, thr6 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    variants.append(("inverse_thresh", thr6))

    # 7. Morphological Gradients
    kernel7 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph_grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel7)
    _, thr7 = cv2.threshold(morph_grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("morph_gradient_thresh", thr7))

    # 8. CLAHE + Sobel + Threshold
    sobelx = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_8U, 0, 1, ksize=3)
    sobel = cv2.bitwise_or(sobelx, sobely)
    clahe_sobel = clahe.apply(sobel)
    _, thr8 = cv2.threshold(clahe_sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("clahe_sobel_thresh", thr8))

    # 9. Adaptive Threshold + Morph Close
    thr9 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 15, 5)
    kernel9 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph9 = cv2.morphologyEx(thr9, cv2.MORPH_CLOSE, kernel9)
    variants.append(("adaptive_thresh_morph_close", morph9))

    # 10. Gaussian Blur + Otsu Threshold
    blur10 = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thr10 = cv2.threshold(blur10, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("blur_otsu_thresh", thr10))

    # 11. Canny with Auto thresholds
    v = np.median(gray)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edges11 = cv2.Canny(gray, lower, upper)
    variants.append(("auto_canny", edges11))

    # 12. CLAHE + Adaptive Threshold (Gaussian)
    clahe12 = clahe.apply(gray)
    thr12 = cv2.adaptiveThreshold(clahe12, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 15, 2)
    variants.append(("clahe_adaptive_gaussian_thresh", thr12))

    return variants


def find_quads(img, edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_quads = []
    for cnt in contours:
        quad = validate_quad(cnt, edges, img.shape)
        if quad is not None:
            valid_quads.append(quad)
    return valid_quads


def detect_best_quad(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    candidates = []

    variants = preprocess_variants(gray)
    for name, edges in variants:
        quads = find_quads(image, edges)
        for q in quads:
            score = score_quad(q, image.shape)
            candidates.append((score, q))

    if not candidates:
        # Fallback: full image rectangle
        h, w = image.shape[:2]
        full_rect = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
        return full_rect

    # En yüksek skorlu quadi seç
    best = max(candidates, key=lambda x: x[0])[1]
    return order_points(best.reshape(4, 2))


class PerspectiveCorrection(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def run(self):
        img_obj = Image.get_frame(img=self.image, redis_db=self.redis_db)
        if img_obj is None or img_obj.value is None:
            raise ValueError("No input image provided or failed to load.")


        img_np = img_obj.value
        if img_np.dtype != np.uint8:
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)

        best_quad = detect_best_quad(img_np)
        warped = four_point_transform(img_np, best_quad)

        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)

        return build_response(context=self)


if __name__ == "__main__":
    Executor(sys.argv[1]).run()
