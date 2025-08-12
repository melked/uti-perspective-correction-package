import os
import sys
import cv2
import numpy as np
from PIL import Image as PILImage

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
                 good_feature_min_dist=20,
                 block_size_adaptive_thresh=11,
                 c_adaptive_thresh=2):
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
        self.block_size_adaptive_thresh = block_size_adaptive_thresh
        self.c_adaptive_thresh = c_adaptive_thresh


def analyze_brightness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray.mean() / 255.0  # 0-1 arası parlaklık


def adaptive_gamma(img, params: Params):
    brightness = analyze_brightness(img)
    # Basit adaptif gamma ayarı
    if brightness < 0.3:
        return min(3.0, params.gamma_target * 2.0)  # Karanlıksa gamma yüksel
    elif brightness > 0.7:
        return max(0.3, params.gamma_target * 0.6)  # Çok parlaksa düşür
    else:
        return params.gamma_target  # Normal durum


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def preprocess(img, params: Params):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=params.clahe_clip, tileGridSize=params.clahe_grid)
    enhanced = clahe.apply(gray)

    gamma_val = adaptive_gamma(img, params)
    table = np.array([(i / 255.0) ** (1.0 / gamma_val) * 255 for i in range(256)]).astype(np.uint8)
    gamma_corrected = cv2.LUT(enhanced, table)

    blurred = cv2.GaussianBlur(gamma_corrected, (5, 5), 0)
    return blurred


def detect_corners(img, params: Params):
    # Try Shi-Tomasi corners first
    corners = cv2.goodFeaturesToTrack(img,
                                      maxCorners=params.max_good_features,
                                      qualityLevel=params.good_feature_quality,
                                      minDistance=params.good_feature_min_dist)

    if corners is not None and len(corners) >= 4:
        corners = np.squeeze(corners)
        if len(corners) > 4:
            # Select the 4 corners furthest from the center
            center = corners.mean(axis=0)
            dists = np.linalg.norm(corners - center, axis=1)
            idxs = np.argsort(dists)[-4:]
            corners = corners[idxs]
        return order_points(corners)
    else:
        # If Shi-Tomasi fails, try contour-based detection
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, params.approx_poly_epsilon_ratio * peri, True)
            if len(approx) == 4 and cv2.contourArea(approx) > params.min_contour_area:
                return order_points(approx.reshape(4, 2))
        return None


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
    pre = preprocess(img, params)

    # Try adaptive thresholding first
    thresh = cv2.adaptiveThreshold(pre, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, params.block_size_adaptive_thresh, params.c_adaptive_thresh)
    edges = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((params.morph_kernel_size, params.morph_kernel_size), np.uint8))

    corners = detect_corners(edges, params)

    if corners is None:
        # If adaptive thresholding fails, try Canny edge detection
        edges = cv2.Canny(pre, params.canny_min, params.canny_max)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((params.morph_kernel_size, params.morph_kernel_size), np.uint8))
        corners = detect_corners(edges, params)

    if corners is None:
        raise ValueError("Belge köşeleri bulunamadı.")

    warped = four_point_transform(img, corners)
    return PILImage.fromarray(warped)


class PerspectiveCorrection(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")
        # Parametre opsiyonel olarak dışarıdan alınabilir
        params_data = self.request.get_param("params", None)
        if params_data is not None:
            self.params = Params(**params_data)
        else:
            self.params = Params()

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

        result_img = correct_perspective(img_np, self.params)
        img.value = np.array(result_img)
        self.image = Image.set_frame(img=img, package_uID=self.uID, redis_db=self.redis_db)
        return build_response(context=self)


if __name__ == "__main__":
    Executor(sys.argv[1]).run()