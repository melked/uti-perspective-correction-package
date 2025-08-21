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


# --- 🔹 PREPROCESS VARIANTS ---
def preprocess_variants(img, params: Params):
    variants = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. CLAHE + Blur
    clahe = cv2.createCLAHE(clipLimit=params.clahe_clip, tileGridSize=params.clahe_grid)
    enhanced = clahe.apply(gray)
    variants.append(cv2.GaussianBlur(enhanced, params.blur_ksize, 0))

    # 2. Gamma correction (farklı ışık koşulları için)
    for gamma in [0.7, 1.3]:
        gamma_img = np.array(255 * (gray / 255.0) ** (1.0 / gamma), dtype='uint8')
        variants.append(cv2.GaussianBlur(gamma_img, (3, 3), 0))

    # 3. Unsharp mask (keskinleştirme)
    blur = cv2.GaussianBlur(gray, (9, 9), 10)
    unsharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    variants.append(unsharp)

    return variants


# --- 🔹 EDGE VARIANTS ---
def edge_variants(img, params: Params):
    edges = []

    # 1. Canny (otomatik threshold)
    if params.canny_min is None or params.canny_max is None:
        v = np.median(img)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
    else:
        lower, upper = params.canny_min, params.canny_max
    edges.append(cv2.Canny(img, lower, upper))

    # 2. Adaptive Threshold
    edges.append(cv2.adaptiveThreshold(img, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2))

    # 3. Sobel
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely)
    sobel = np.uint8(np.clip(sobel, 0, 255))
    edges.append(sobel)

    return edges


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


# --- 🔹 QUAD SCORING ---
def evaluate_quad(corners, img_shape):
    """Dörtgenin düzgünlüğüne puan verir."""
    if corners is None:
        return 0

    (h, w) = img_shape[:2]
    rect = order_points(corners)

    # Alan oranı (belge ekranın %10'undan küçükse düşük puan)
    area = cv2.contourArea(rect.astype(np.int32))
    score_area = min(area / (w * h), 1.0)

    # Açılar (90 dereceye yakınsa iyi)
    def angle(p1, p2, p3):
        v1, v2 = p1 - p2, p3 - p2
        cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

    angles = []
    for i in range(4):
        angles.append(angle(rect[i], rect[(i + 1) % 4], rect[(i + 2) % 4]))
    score_angle = 1 - (np.std(angles) / 90)  # ne kadar dikdörtgene yakınsa o kadar iyi

    return 0.7 * score_area + 0.3 * score_angle


# --- 🔹 ANA FONKSİYON ---
def correct_perspective_auto(img, params: Params):
    candidates = []

    for pre in preprocess_variants(img, params):
        for edge in edge_variants(pre, params):
            edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE,
                                    np.ones((params.morph_kernel_size, params.morph_kernel_size), np.uint8))
            corners = detect_corners(edge, params)
            if corners is not None:
                warped = four_point_transform(img, corners)
                score = evaluate_quad(corners, warped.shape)
                candidates.append((score, warped))

    if not candidates:
        raise ValueError("Belge köşeleri bulunamadı.")

    # En iyi skorlu belgeyi döndür
    candidates.sort(key=lambda x: x[0], reverse=True)
    return PILImage.fromarray(candidates[0][1])


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

        result_img = correct_perspective_auto(img_np, self.params)
        img.value = np.array(result_img)
        self.image = Image.set_frame(img=img, package_uID=self.uID, redis_db=self.redis_db)
        return build_response(context=self)


if __name__ == "__main__":
    Executor(sys.argv[1]).run()
