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
    """A4/belge formatları için optimize edilmiş 6 yöntemle köşe algılama"""
    h, w = img.shape[:2]
    min_area = (h * w) * 0.1  # Minimum %10 alan
    methods = []

    # Method 1: Document-specific GoodFeatures
    corners = cv2.goodFeaturesToTrack(img, maxCorners=16, qualityLevel=0.01, minDistance=min(h, w) // 20)
    if corners is not None and len(corners) >= 4:
        corners = np.squeeze(corners)
        # En köşe pozisyonundaki 4 noktayı seç
        corner_scores = []
        for c in corners:
            x, y = c
            # Köşelere yakınlık skoru (0-1)
            score = min(x / w, (w - x) / w) + min(y / h, (h - y) / h) + max(x / w, (w - x) / w) + max(y / h,
                                                                                                      (h - y) / h)
            corner_scores.append(score)
        best_indices = np.argsort(corner_scores)[-4:]
        methods.append(corners[best_indices])

    # Method 2: Rectangle-optimized contours
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Alan ve dikdörtgenlik kontrolü
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
            area = cv2.contourArea(contour)
            if area > min_area:
                # Dikdörtgen yaklaşımı
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)

                # Contour'un dikdörtgene ne kadar yakın olduğunu kontrol et
                hull = cv2.convexHull(contour)
                epsilon = 0.015 * cv2.arcLength(hull, True)  # A4 için optimize
                approx = cv2.approxPolyDP(hull, epsilon, True)

                if 4 <= len(approx) <= 6:  # Dikdörtgen ya da yakın şekil
                    if len(approx) == 4:
                        methods.append(approx.reshape(-1, 2))
                    else:
                        # En uygun 4 köşeyi seç
                        points = approx.reshape(-1, 2)
                        center = np.mean(points, axis=0)
                        dists = np.linalg.norm(points - center, axis=1)
                        methods.append(points[np.argsort(dists)[-4:]])
                    break

    # Method 3: Hough lines for document edges
    lines = cv2.HoughLines(img, 1, np.pi / 180, min(h, w) // 4)
    if lines is not None and len(lines) >= 4:
        # Yatay ve dikey çizgileri ayır
        horizontal, vertical = [], []
        for rho, theta in lines[:20, 0]:  # İlk 20 çizgi
            angle = theta * 180 / np.pi
            if abs(angle) < 10 or abs(angle - 180) < 10:  # Yatay
                horizontal.append((rho, theta))
            elif abs(angle - 90) < 10:  # Dikey
                vertical.append((rho, theta))

        # En az 2 yatay, 2 dikey çizgi varsa kesişimleri hesapla
        if len(horizontal) >= 2 and len(vertical) >= 2:
            intersections = []
            for rho1, theta1 in horizontal[:2]:
                for rho2, theta2 in vertical[:2]:
                    # Çizgi kesişimi hesapla
                    cos1, sin1 = np.cos(theta1), np.sin(theta1)
                    cos2, sin2 = np.cos(theta2), np.sin(theta2)
                    det = cos1 * sin2 - sin1 * cos2
                    if abs(det) > 1e-6:
                        x = (sin2 * rho1 - sin1 * rho2) / det
                        y = (cos1 * rho2 - cos2 * rho1) / det
                        if 0 <= x < w and 0 <= y < h:
                            intersections.append([x, y])

            if len(intersections) >= 4:
                intersections = np.array(intersections)
                # En köşe 4 noktayı seç
                center = np.mean(intersections, axis=0)
                dists = np.linalg.norm(intersections - center, axis=1)
                methods.append(intersections[np.argsort(dists)[-4:]])

    # Method 4: Edge-based corner detection
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    corners = cv2.goodFeaturesToTrack(opened, maxCorners=12, qualityLevel=0.005, minDistance=min(h, w) // 15)
    if corners is not None and len(corners) >= 4:
        corners = np.squeeze(corners)
        center = np.mean(corners, axis=0)
        dists = np.linalg.norm(corners - center, axis=1)
        methods.append(corners[np.argsort(dists)[-4:]])

    # Method 5: Adaptive threshold + contours
    adaptive = cv2.adaptiveThreshold(cv2.GaussianBlur(img, (5, 5), 0), 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    contours, _ = cv2.findContours(adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > min_area:
            epsilon = 0.02 * cv2.arcLength(largest, True)
            approx = cv2.approxPolyDP(largest, epsilon, True)
            if len(approx) >= 4:
                methods.append(approx.reshape(-1, 2)[:4])

    # Method 6: Fallback - A4 oranında köşeler
    if not methods:
        margin = min(h, w) * 0.05  # %5 margin
        a4_corners = np.array([
            [margin, margin],
            [w - margin, margin],
            [w - margin, h - margin],
            [margin, h - margin]
        ])
        methods.append(a4_corners)

    # En iyi yöntemi seç (alan + A4 oranı uygunluğu)
    best_corners = None
    best_score = 0

    for corners in methods:
        try:
            if len(corners) < 4: continue
            ordered = reorder_corners(corners[:4])
            area = cv2.contourArea(ordered.astype(np.float32))

            # A4 oranı kontrolü (√2 ≈ 1.414)
            side_lengths = []
            for i in range(4):
                p1, p2 = ordered[i], ordered[(i + 1) % 4]
                side_lengths.append(np.linalg.norm(p2 - p1))

            # Uzun/kısa kenar oranı A4'e ne kadar yakın
            ratio = max(side_lengths[0], side_lengths[1]) / min(side_lengths[0], side_lengths[1])
            a4_score = 1.0 / (1.0 + abs(ratio - 1.414))  # A4 oranına yakınlık

            # Köşelerin görüntü köşelerine yakınlığı
            corner_distances = []
            image_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]])
            for corner in ordered:
                min_dist = min([np.linalg.norm(corner - ic) for ic in image_corners])
                corner_distances.append(min_dist)
            corner_score = 1.0 / (1.0 + np.mean(corner_distances) / min(h, w))

            score = area * a4_score * corner_score
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