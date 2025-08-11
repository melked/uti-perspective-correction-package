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
    def __init__(self, config=None):
        config = config or {}
        self.clahe_clip = config.get("clahe_clip", 3.0)
        self.clahe_grid = tuple(config.get("clahe_grid", (8, 8)))
        self.blur_ksize = tuple(config.get("blur_ksize", (5, 5)))
        self.morph_kernel_size = config.get("morph_kernel_size", 5)
        self.max_corners = config.get("max_corners", 20)
        self.quality_level = config.get("quality_level", 0.01)
        self.min_distance = config.get("min_distance", 20)
        self.contour_area_thresh = config.get("contour_area_thresh", 1000)
        self.approx_poly_epsilon_ratio = config.get("approx_poly_epsilon_ratio", 0.02)
        # Canny eşikleri opsiyonel, yoksa otomatik hesaplanacak
        self.canny_min = config.get("canny_min", None)
        self.canny_max = config.get("canny_max", None)


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left
    return rect


def preprocess(img, params: Params):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=params.clahe_clip, tileGridSize=params.clahe_grid)
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, params.blur_ksize, 0)
    return blurred


def detect_corners(img, params: Params):
    corners = cv2.goodFeaturesToTrack(img,
                                      maxCorners=params.max_corners,
                                      qualityLevel=params.quality_level,
                                      minDistance=params.min_distance)
    if corners is None or len(corners) < 4:
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, params.approx_poly_epsilon_ratio * peri, True)
            if len(approx) == 4 and cv2.contourArea(approx) > params.contour_area_thresh:
                return order_points(approx.reshape(4, 2))
        return None
    corners = np.squeeze(corners)
    if len(corners) > 4:
        center = corners.mean(axis=0)
        dists = np.linalg.norm(corners - center, axis=1)
        idxs = np.argsort(dists)[-4:]
        corners = corners[idxs]
    return order_points(corners)


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

    # Otomatik Canny eşiği hesaplama (median tabanlı)
    if params.canny_min is None or params.canny_max is None:
        v = np.median(pre)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
    else:
        lower, upper = params.canny_min, params.canny_max

    edges = cv2.Canny(pre, lower, upper)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                             np.ones((params.morph_kernel_size, params.morph_kernel_size), np.uint8))
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
        params_data = self.request.get_param("params", {})
        self.params = Params(params_data)

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
