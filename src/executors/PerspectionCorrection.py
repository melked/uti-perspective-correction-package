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
    """Köşeleri sırala: üst-sol, üst-sağ, alt-sağ, alt-sol"""
    center = np.mean(corners, axis=0)
    ordered = np.zeros((4, 2), dtype=corners.dtype)
    for c in corners:
        if c[0] < center[0] and c[1] < center[1]:
            ordered[0] = c  # üst-sol
        elif c[0] > center[0] and c[1] < center[1]:
            ordered[1] = c  # üst-sağ
        elif c[0] > center[0] and c[1] > center[1]:
            ordered[2] = c  # alt-sağ
        else:
            ordered[3] = c  # alt-sol
    return ordered


def preprocess(img, clahe_clip=3.0, clahe_grid=(8, 8), gamma=1.0, blur_ksize=5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    enhanced = clahe.apply(gray)

    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    gamma_corrected = cv2.LUT(enhanced, table)

    return cv2.GaussianBlur(gamma_corrected, (blur_ksize, blur_ksize), 0)


def detect_corners_combined(img):
    """Farklı yöntemleri birleştirip en iyi 4 köşeyi tespit eder."""
    h, w = img.shape[:2]
    min_area = h * w * 0.1
    candidates = []

    # Yöntem 1: goodFeaturesToTrack ile köşe bul
    corners = cv2.goodFeaturesToTrack(img, maxCorners=16, qualityLevel=0.01, minDistance=min(h, w) // 20)
    if corners is not None and len(corners) >= 4:
        corners = np.squeeze(corners)
        # Köşelere yakın 4 noktayı seç
        scores = []
        for c in corners:
            x, y = c
            score = min(x, w - x) + min(y, h - y)  # köşeye yakınlık skoru
            scores.append(score)
        idxs = np.argsort(scores)[-4:]
        candidates.append(corners[idxs])

    # Yöntem 2: Kontur bul, en büyük 3 kontur içinde dikdörtgeni kontrol et
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
        if cv2.contourArea(cnt) > min_area:
            hull = cv2.convexHull(cnt)
            epsilon = 0.02 * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)
            if len(approx) == 4:
                candidates.append(approx.reshape(4, 2))
                break

    # Yöntem 3: Hough Lines ile çizgi kesişimleri
    lines = cv2.HoughLines(img, 1, np.pi / 180, int(min(h, w) / 4))
    if lines is not None:
        horizontal, vertical = [], []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta) % 180
            if angle < 10 or angle > 170:
                horizontal.append((rho, theta))
            elif 80 < angle < 100:
                vertical.append((rho, theta))
        if len(horizontal) >= 2 and len(vertical) >= 2:
            intersections = []
            for (rho1, theta1) in horizontal[:2]:
                for (rho2, theta2) in vertical[:2]:
                    cos1, sin1 = np.cos(theta1), np.sin(theta1)
                    cos2, sin2 = np.cos(theta2), np.sin(theta2)
                    denom = cos1 * sin2 - sin1 * cos2
                    if abs(denom) > 1e-6:
                        x = (sin2 * rho1 - sin1 * rho2) / denom
                        y = (cos1 * rho2 - cos2 * rho1) / denom
                        if 0 <= x < w and 0 <= y < h:
                            intersections.append([x, y])
            if len(intersections) >= 4:
                intersections = np.array(intersections)
                center = np.mean(intersections, axis=0)
                dists = np.linalg.norm(intersections - center, axis=1)
                candidates.append(intersections[np.argsort(dists)[-4:]])

    # En iyi adayları seç (en büyük alan ve A4 oranına yakın)
    best_corners, best_score = None, 0
    for c in candidates:
        if len(c) != 4:
            continue
        c = reorder_corners(c)
        area = cv2.contourArea(c.astype(np.float32))
        if area < min_area:
            continue
        side_lens = [np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)]
        ratio = max(side_lens) / min(side_lens)
        a4_score = 1 / (1 + abs(ratio - 1.414))  # A4 oranına yakınlık
        score = area * a4_score
        if score > best_score:
            best_score = score
            best_corners = c

    # Fallback: Görüntü köşeleri (marjinle)
    if best_corners is None:
        margin = min(h, w) * 0.05
        best_corners = np.array([
            [margin, margin],
            [w - margin, margin],
            [w - margin, h - margin],
            [margin, h - margin]
        ])

    return best_corners


def correct_perspective(img, params):
    pre = preprocess(img, params["clahe_clip"], params["clahe_grid"], params["gamma"], params["blur_ksize"])
    edges = cv2.Canny(pre, params["canny_min"], params["canny_max"])

    corners = detect_corners_combined(edges)
    if corners is None:
        raise ValueError("Yeterli köşe bulunamadı")

    h, w = img.shape[:2]
    min_dim = min(h, w)
    new_w, new_h = int(min_dim), int(min_dim * params["output_ratio"])

    dst = np.float32([[0, 0], [new_w, 0], [new_w, new_h], [0, new_h]])
    M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    warped = cv2.warpPerspective(img, M, (new_w, new_h))

    return PILImage.fromarray(warped)


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
            "output_ratio": self.request.get_param("output_ratio") or 0.707,
        }
        self.image = self.request.get_param("inputImage")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def run(self):
        img = Image.get_frame(img=self.image, redis_db=self.redis_db)
        img_np = img.value

        if img_np.dtype != np.uint8:
            img_np = (img_np * 255).astype(np.uint8) if img_np.max() <= 1.0 else img_np.astype(np.uint8)

        result = correct_perspective(img_np, self.params)

        img.value = np.array(result)
        self.image = Image.set_frame(img=img, package_uID=self.uID, redis_db=self.redis_db)
        return build_response(context=self)


if __name__ == "__main__":
    Executor(sys.argv[1]).run()
