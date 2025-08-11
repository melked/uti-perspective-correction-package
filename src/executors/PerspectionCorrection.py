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
    """7 farklı yöntemle köşe algılar ve en iyisini seçer"""
    h, w = img.shape[:2]
    methods = []

    # Method 1: GoodFeaturesToTrack (orijinal)
    corners = cv2.goodFeaturesToTrack(img, maxCorners=max_corners, qualityLevel=quality, minDistance=min_distance)
    if corners is not None and len(corners) >= 4:
        corners = np.squeeze(corners)
        if len(corners) > 4:
            center = np.mean(corners, axis=0)
            dists = np.linalg.norm(corners - center, axis=1)
            corners = corners[np.argsort(dists)[-4:]]
        methods.append(corners[:4])

    # Method 2: GoodFeaturesToTrack (daha hassas)
    corners = cv2.goodFeaturesToTrack(img, maxCorners=20, qualityLevel=0.005, minDistance=15)
    if corners is not None and len(corners) >= 4:
        corners = np.squeeze(corners)
        center = np.mean(corners, axis=0)
        dists = np.linalg.norm(corners - center, axis=1)
        corners = corners[np.argsort(dists)[-4:]]
        methods.append(corners)

    # Method 3: Contour + ApproxPolyDP (gevşek)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 500:
            epsilon = 0.02 * cv2.arcLength(largest, True)
            approx = cv2.approxPolyDP(largest, epsilon, True)
            if len(approx) >= 4:
                methods.append(approx.reshape(-1, 2)[:4])

    # Method 4: Contour + ApproxPolyDP (sıkı)
    if contours:
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
            if cv2.contourArea(contour) > 1000:
                hull = cv2.convexHull(contour)
                epsilon = 0.01 * cv2.arcLength(hull, True)
                approx = cv2.approxPolyDP(hull, epsilon, True)
                if len(approx) >= 4:
                    methods.append(approx.reshape(-1, 2)[:4])
                    break

    # Method 5: Hough Lines kesişimleri
    lines = cv2.HoughLinesP(img, 1, np.pi / 180, 80, minLineLength=w // 4, maxLineGap=20)
    if lines is not None and len(lines) >= 3:
        intersections = []
        lines = lines.reshape(-1, 4)
        for i in range(len(lines)):
            for j in range(i + 1, min(len(lines), i + 10)):  # Sadece yakın çizgiler
                x1, y1, x2, y2 = lines[i]
                x3, y3, x4, y4 = lines[j]
                denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if abs(denom) > 1e-6:
                    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
                    x = x1 + t * (x2 - x1)
                    y = y1 + t * (y2 - y1)
                    if 0 <= x < w and 0 <= y < h:
                        intersections.append([x, y])
        if len(intersections) >= 4:
            intersections = np.array(intersections)
            center = np.mean(intersections, axis=0)
            dists = np.linalg.norm(intersections - center, axis=1)
            methods.append(intersections[np.argsort(dists)[-4:]])

    # Method 6: Corner dilation (morfolojik)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(img, kernel, iterations=1)
    corners = cv2.goodFeaturesToTrack(dilated, maxCorners=15, qualityLevel=0.01, minDistance=20)
    if corners is not None and len(corners) >= 4:
        corners = np.squeeze(corners)
        center = np.mean(corners, axis=0)
        dists = np.linalg.norm(corners - center, axis=1)
        methods.append(corners[np.argsort(dists)[-4:]])

    # Method 7: Grid-based corner detection (son çare)
    if not methods:
        grid_corners = []
        for i in [0.1, 0.9]:
            for j in [0.1, 0.9]:
                grid_corners.append([int(i * w), int(j * h)])
        methods.append(np.array(grid_corners))

    # En iyi yöntemi seç (alan + şekil uygunluğu)
    best_corners = None
    best_score = 0

    for corners in methods:
        try:
            if len(corners) < 4: continue
            ordered = reorder_corners(corners[:4])
            area = cv2.contourArea(ordered.astype(np.float32))

            # Dikdörtgen uygunluk skoru
            side_lengths = []
            for i in range(4):
                p1, p2 = ordered[i], ordered[(i + 1) % 4]
                side_lengths.append(np.linalg.norm(p2 - p1))

            # Karşılıklı kenarların benzerliği
            ratio1 = min(side_lengths[0], side_lengths[2]) / max(side_lengths[0], side_lengths[2])
            ratio2 = min(side_lengths[1], side_lengths[3]) / max(side_lengths[1], side_lengths[3])
            shape_score = (ratio1 + ratio2) / 2

            score = area * shape_score
            if score > best_score:
                best_score = score
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