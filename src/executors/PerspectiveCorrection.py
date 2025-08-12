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
        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe_obj.apply(img)

    def unsharp_mask(img):
        gaussian = cv2.GaussianBlur(img, (9, 9), 10.0)
        return cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)

    def preprocess_variants(img):
        variants = []

        v1 = clahe(img)
        edges1 = cv2.Canny(v1, 50, 150)
        variants.append(("normal", edges1))

        v2 = unsharp_mask(clahe(img))
        edges2 = cv2.Canny(v2, 50, 150)
        variants.append(("agresif", edges2))

        v3 = cv2.GaussianBlur(img, (5, 5), 0)
        thr = cv2.adaptiveThreshold(v3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
        variants.append(("yumusak", thr))

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
        center = np.array([w/2, h/2])

        area = cv2.contourArea(cnt)
        area_score = 1 - abs(area - 0.25*w*h) / (0.25*w*h)

        widths = [np.linalg.norm(pts[i] - pts[(i+1)%4]) for i in range(4)]
        height_diff = abs(widths[0] - widths[2]) / max(widths[0], widths[2])
        width_diff = abs(widths[1] - widths[3]) / max(widths[1], widths[3])
        rect_score = 1 - (height_diff + width_diff) / 2

        cnt_center = np.mean(pts, axis=0)
        dist_center = np.linalg.norm(cnt_center - center)
        max_dist = np.linalg.norm(np.array([w/2, h/2]))
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
