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


def reorder_corners(corners):
    """Köşeleri düzenler: üst-sol, üst-sağ, alt-sağ, alt-sol"""
    center = np.mean(corners, axis=0)
    ordered = np.zeros((4, 2), dtype=corners.dtype)

    for c in corners:
        if c[0] < center[0] and c[1] < center[1]:
            ordered[0] = c  # upper-left
        elif c[0] > center[0] and c[1] < center[1]:
            ordered[1] = c  # upper-right
        elif c[0] > center[0] and c[1] > center[1]:
            ordered[2] = c  # lower-right
        else:
            ordered[3] = c  # lower-left
    return ordered


def preprocess(img, clahe_clip=3.0, clahe_grid=(8, 8), gamma=1.0, blur_ksize=5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    enhanced = clahe.apply(gray)

    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    gamma_corrected = cv2.LUT(enhanced, table)

    blurred = cv2.GaussianBlur(gamma_corrected, (blur_ksize, blur_ksize), 0)
    return blurred


def detect_corners_combined(img, max_corners=10, quality=0.01, min_distance=30):
    """Birden fazla yöntemle köşe algılar ve en iyisini seçer"""
    methods = []

    # Method 1: GoodFeaturesToTrack (orijinal)
    corners1 = cv2.goodFeaturesToTrack(img, maxCorners=max_corners, qualityLevel=quality, minDistance=min_distance)
    if corners1 is not None and len(corners1) >= 4:
        corners1 = np.squeeze(corners1)
        if len(corners1) > 4:
            center = np.mean(corners1, axis=0)
            dists = np.linalg.norm(corners1 - center, axis=1)
            corners1 = corners1[np.argsort(dists)[-4:]]
        methods.append(("goodfeatures", corners1[:4]))

    # Method 2: Contour + ApproxPolyDP
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 1000:
            hull = cv2.convexHull(largest)
            epsilon = 0.02 * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)
            if len(approx) >= 4:
                corners2 = approx.reshape(-1, 2)[:4]
                methods.append(("contour", corners2))

    # Method 3: Hough Lines (basitleştirilmiş)
    lines = cv2.HoughLinesP(img, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    if lines is not None and len(lines) >= 4:
        # Sadece en uzun 4 çizgiyi al ve köşe olarak uç noktalarını kullan
        lines = lines.reshape(-1, 4)
        lengths = [np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) for x1, y1, x2, y2 in lines]
        longest_indices = np.argsort(lengths)[-4:]
        corners3 = []
        for idx in longest_indices:
            x1, y1, x2, y2 = lines[idx]
            corners3.extend([[x1, y1], [x2, y2]])
        if len(corners3) >= 4:
            corners3 = np.array(corners3)
            # En dış 4 köşeyi seç
            center = np.mean(corners3, axis=0)
            dists = np.linalg.norm(corners3 - center, axis=1)
            corners3 = corners3[np.argsort(dists)[-4:]]
            methods.append(("hough", corners3))

    # En iyi yöntemi seç (en büyük alan kriteri)
    best_corners = None
    best_area = 0

    for name, corners in methods:
        try:
            ordered = reorder_corners(corners)
            area = cv2.contourArea(ordered.astype(np.float32))
            if area > best_area:
                best_area = area
                best_corners = ordered
        except:
            continue

    return best_corners


def correct_perspective(
        img,
        clahe_clip=3.0,
        clahe_grid=(8, 8),
        gamma=1.0,
        blur_ksize=5,
        canny_min=50,
        canny_max=150,
        max_corners=10,
        quality_level=0.01,
        min_distance=30,
        output_ratio=0.707,
        intermediate=False,
):
    pre = preprocess(img, clahe_clip, clahe_grid, gamma, blur_ksize)
    edges = cv2.Canny(pre, canny_min, canny_max)

    # Birleşik köşe algılama
    corners = detect_corners_combined(edges, max_corners, quality_level, min_distance)

    if corners is None:
        raise ValueError("Yeterli köşe bulunamadı")

    h, w = img.shape[:2]
    min_dim = min(h, w)
    new_w, new_h = int(min_dim), int(min_dim * output_ratio)

    dst = np.float32([[0, 0], [new_w, 0], [new_w, new_h], [0, new_h]])
    M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    warped = cv2.warpPerspective(img, M, (new_w, new_h))

    if intermediate:
        return (PILImage.fromarray(pre), PILImage.fromarray(edges), PILImage.fromarray(warped))
    else:
        return (PILImage.fromarray(warped),)


class PerspectiveCorrection(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.params = {
            "clahe_clip": self.request.get_param("clahe_clip") or 3.0,
            "clahe_grid": self.request.get_param("clahe_grid") or (8, 8),
            "gamma": self.request.get_param("gamma") or 1.0,
            "blur_ksize": self.request.get_param("blur_ksize") or 5,
            "canny_min": self.request.get_param("canny_min") or 50,
            "canny_max": self.request.get_param("canny_max") or 150,
            "max_corners": self.request.get_param("max_corners") or 10,
            "quality_level": self.request.get_param("quality_level") or 0.01,
            "min_distance": self.request.get_param("min_distance") or 30,
            "output_ratio": self.request.get_param("output_ratio") or 0.707,
            "intermediate": self.request.get_param("intermediate") or False,
        }
        self.image = self.request.get_param("inputImage")

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

        result = correct_perspective(
            img_np,
            clahe_clip=self.params["clahe_clip"],
            clahe_grid=self.params["clahe_grid"],
            gamma=self.params["gamma"],
            blur_ksize=self.params["blur_ksize"],
            canny_min=self.params["canny_min"],
            canny_max=self.params["canny_max"],
            max_corners=self.params["max_corners"],
            quality_level=self.params["quality_level"],
            min_distance=self.params["min_distance"],
            output_ratio=self.params["output_ratio"],
            intermediate=self.params["intermediate"],
        )

        img.value = np.array(result[-1])
        self.image = Image.set_frame(img=img, package_uID=self.uID, redis_db=self.redis_db)

        return build_response(context=self)


if __name__ == "__main__":
    Executor(sys.argv[1]).run()