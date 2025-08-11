import os
import sys
import cv2
import numpy as np
from PIL import Image as PILImage
from sklearn.cluster import DBSCAN

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveCorrection.src.utils.response import build_response
from components.PerspectiveCorrection.src.models.PackageModel import PackageModel


class Params:
    def __init__(self,
                 clahe_clip=3.0,
                 clahe_grid=(8, 8),
                 gamma_target=0.5,
                 canny_min=50,
                 canny_max=150,
                 morph_kernel_size=5,
                 min_contour_area=1000,
                 approx_poly_epsilon_ratio=0.02,
                 max_good_features=20,
                 good_feature_quality=0.01,
                 good_feature_min_dist=20):
        self.clahe_clip = clahe_clip
        self.clahe_grid = clahe_grid
        self.gamma_target = gamma_target
        self.canny_min = canny_min
        self.canny_max = canny_max
        self.morph_kernel_size = morph_kernel_size
        self.min_contour_area = min_contour_area
        self.approx_poly_epsilon_ratio = approx_poly_epsilon_ratio
        self.max_good_features = max_good_features
        self.good_feature_quality = good_feature_quality
        self.good_feature_min_dist = good_feature_min_dist


def angle_between(p1, p2, p3):
    a = np.array(p1) - np.array(p2)
    b = np.array(p3) - np.array(p2)
    cos_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def filter_quadrilaterals(contours, min_area=1000, angle_tol=15):
    quads = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            angles = [angle_between(pts[i - 1], pts[i], pts[(i + 1) % 4]) for i in range(4)]
            if all(80 - angle_tol <= a <= 100 + angle_tol for a in angles):
                quads.append((approx, area))
    return sorted(quads, key=lambda x: x[1], reverse=True)


def analyze_brightness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray.mean() / 255.0


def adaptive_gamma(img, params: Params):
    brightness = analyze_brightness(img)
    if brightness < 0.3:
        return min(3.0, params.gamma_target * 2.0)
    elif brightness > 0.7:
        return max(0.3, params.gamma_target * 0.6)
    else:
        return params.gamma_target


def preprocess(img, params: Params):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=params.clahe_clip, tileGridSize=params.clahe_grid)
    enhanced = clahe.apply(gray)
    gamma_val = adaptive_gamma(img, params)
    table = np.array([(i / 255.0) ** (1.0 / gamma_val) * 255 for i in range(256)]).astype(np.uint8)
    gamma_corrected = cv2.LUT(enhanced, table)
    blurred = cv2.GaussianBlur(gamma_corrected, (5, 5), 0)
    return blurred


def detect_document_corners(img, params: Params):
    pre = preprocess(img, params)
    median_val = np.median(pre)
    lower = int(max(0, 0.66 * median_val))
    upper = int(min(255, 1.33 * median_val))
    edges = cv2.Canny(pre, lower, upper)
    kernel = np.ones((params.morph_kernel_size, params.morph_kernel_size), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quads = filter_quadrilaterals(contours, min_area=params.min_contour_area)
    if not quads:
        raise ValueError("Belge köşeleri bulunamadı.")
    biggest_approx = quads[0][0]
    return order_points(biggest_approx.reshape(4, 2))


def four_point_transform(img, pts):
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
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return warped


def correct_perspective(img, params: Params):
    corners = detect_document_corners(img, params)
    warped = four_point_transform(img, corners)
    return PILImage.fromarray(warped)


class PerspectiveCorrection(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")
        params_data = self.request.get_param("params", None)
        self.params = Params(**params_data) if params_data else Params()

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def run(self):
        img_obj = Image.get_frame(img=self.image, redis_db=self.redis_db)
        img_np = img_obj.value
        if img_np.dtype != np.uint8:
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)

        result_img = correct_perspective(img_np, self.params)
        img_obj.value = np.array(result_img)
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)
        return build_response(context=self)


if __name__ == "__main__":
    Executor(sys.argv[1]).run()
