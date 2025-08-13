import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveTransformation.src.utils.response import build_response
from components.PerspectiveTransformation.src.models.PackageModel import PackageModel


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def correct_perspective_advanced(image, params=None):
    gray_orig = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def clahe(img):
        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe_obj.apply(img)

    def unsharp_mask(img):
        gaussian = cv2.GaussianBlur(img, (9, 9), 10.0)
        return cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)

    def preprocess_variants(img):
        variants = []

        # Canny eşik kombinasyonları
        canny_params = [(30, 100), (50, 150), (100, 200)]

        # Gaussian kernel boyutları
        blur_kernels = [(3, 3), (5, 5), (7, 7)]

        # 1️⃣ Normal CLAHE + Canny (farklı eşikler)
        for low, high in canny_params:
            v1 = clahe(img)
            edges1 = cv2.Canny(v1, low, high)
            variants.append((f"normal_{low}_{high}", edges1))

        # 2️⃣ Agresif CLAHE + Unsharp + Canny
        for low, high in canny_params:
            v2 = unsharp_mask(clahe(img))
            edges2 = cv2.Canny(v2, low, high)
            variants.append((f"agresif_{low}_{high}", edges2))

        # 3️⃣ Yumuşak Gaussian Blur + Adaptive Threshold (farklı kernel boyutları)
        for k in blur_kernels:
            v3 = cv2.GaussianBlur(img, k, 0)
            thr3 = cv2.adaptiveThreshold(v3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
            variants.append((f"yumusak_{k[0]}", thr3))

        # 4️⃣ Gamma Correction + CLAHE + Canny
        for gamma_val in [0.6, 0.8, 1.2]:
            v4 = clahe(img)
            gamma_corrected = np.array(np.power(v4 / 255.0, gamma_val) * 255, dtype=np.uint8)
            edges4 = cv2.Canny(gamma_corrected, 50, 150)
            variants.append((f"gamma_{gamma_val}", edges4))

        # 5️⃣ CLAHE + Otsu Threshold
        v5 = clahe(img)
        otsu_thr = cv2.threshold(v5, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        variants.append(("otsu", otsu_thr))

        # 6️⃣ Morphological Gradient (farklı kernel boyutları)
        for size in [3, 5]:
            v6 = clahe(img)
            morph = cv2.morphologyEx(v6, cv2.MORPH_GRADIENT, np.ones((size, size), np.uint8))
            variants.append((f"morph_gradient_{size}", morph))

        # 7️⃣ Bilateral Filter + Canny (farklı parametreler)
        for d in [7, 9]:
            v7 = cv2.bilateralFilter(img, d, 75, 75)
            edges7 = cv2.Canny(v7, 50, 150)
            variants.append((f"bilateral_{d}", edges7))

        # 8️⃣ Top-hat Morphology + Canny
        for size in [3, 5]:
            kernel = np.ones((size, size), np.uint8)
            tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
            edges8 = cv2.Canny(tophat, 50, 150)
            variants.append((f"tophat_{size}", edges8))

        return variants

    def validate_contour(cnt, edges, img_shape):
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            return None
        if not cv2.isContourConvex(approx):
            return None
        area = cv2.contourArea(approx)
        h, w = img_shape[:2]
        if area < 0.01 * w * h or area > 0.9 * w * h:
            return None
        pts = approx.reshape(4, 2)
        widths = [np.linalg.norm(pts[i] - pts[(i+1) % 4]) for i in range(4)]
        min_w, max_w = min(widths), max(widths)
        if min_w / max_w < 0.2:
            return None
        mask = np.zeros_like(edges)
        cv2.drawContours(mask, [approx], -1, 255, 2)
        overlap = cv2.countNonZero(cv2.bitwise_and(mask, edges))
        total_edge = cv2.countNonZero(mask)
        if total_edge == 0 or (overlap / total_edge) < 0.7:
            return None
        return approx

    def score_contour(cnt, img_shape):
        pts = cnt.reshape(4, 2)
        h, w = img_shape[:2]
        center = np.array([w / 2, h / 2])
        area = cv2.contourArea(cnt)
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

    variants = preprocess_variants(gray_orig)
    candidates = []

    for name, processed in variants:
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            valid = validate_contour(cnt, processed, image.shape)
            if valid is not None:
                score = score_contour(valid, image.shape)
                candidates.append((score, valid))

    if not candidates:
        thr = cv2.threshold(gray_orig, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            biggest = max(contours, key=cv2.contourArea)
            peri = cv2.arcLength(biggest, True)
            approx = cv2.approxPolyDP(biggest, 0.02 * peri, True)
            if len(approx) == 4:
                candidates.append((0, approx))

    if not candidates:
        return image

    best = max(candidates, key=lambda x: x[0])[1]
    rect = order_points(best.reshape(4, 2))
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


class PerspectiveCorrection(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")
        self.params = self.request.get_param("params", None)

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def run(self):
        img = Image.get_frame(img=self.image, redis_db=self.redis_db)
        img_np = img.value
        if img_np.dtype != np.uint8:
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)
        result_img = correct_perspective_advanced(img_np, self.params)
        img.value = np.array(result_img)
        self.image = Image.set_frame(img=img, package_uID=self.uID, redis_db=self.redis_db)
        return build_response(context=self)

if __name__ == "__main__":
    Executor(sys.argv[1]).run()
