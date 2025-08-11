import os
import sys
import cv2
import numpy as np
from itertools import combinations
from PIL import Image as PILImage

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveCorrection.src.utils.response import build_response
from components.PerspectiveCorrection.src.models.PackageModel import PackageModel


def preprocess(img, clahe_clip, clahe_grid, gamma, blur_ksize):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    enhanced = clahe.apply(gray)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    gamma_corrected = cv2.LUT(enhanced, table)
    blurred = cv2.GaussianBlur(gamma_corrected, (blur_ksize, blur_ksize), 0)
    return blurred


def hough_intersections(edge_img):
    lines = cv2.HoughLinesP(edge_img, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=10)
    if lines is None:
        return None
    lines = lines[:, 0, :]
    points = []
    for (x1, y1, x2, y2) in lines:
        points.append((x1, y1))
        points.append((x2, y2))
    intersections = []
    for (p1, p2), (p3, p4) in combinations(points, 2):
        denom = (p1[0] - p2[0])*(p3[1] - p4[1]) - (p1[1] - p2[1])*(p3[0] - p4[0])
        if denom == 0:
            continue
        x = ((p1[0]*p2[1] - p1[1]*p2[0])*(p3[0]-p4[0]) - (p1[0]-p2[0])*(p3[0]*p4[1] - p3[1]*p4[0])) / denom
        y = ((p1[0]*p2[1] - p1[1]*p2[0])*(p3[1]-p4[1]) - (p1[1]-p2[1])*(p3[0]*p4[1] - p3[1]*p4[0])) / denom
        if 0 <= x < edge_img.shape[1] and 0 <= y < edge_img.shape[0]:
            intersections.append([x, y])
    if len(intersections) < 4:
        return None
    # Kümeleme ile benzer noktaları grupla
    intersections = np.array(intersections)
    clusters = []
    taken = np.zeros(len(intersections), dtype=bool)
    for i, pt in enumerate(intersections):
        if taken[i]:
            continue
        cluster = [pt]
        taken[i] = True
        for j in range(i+1, len(intersections)):
            if taken[j]:
                continue
            if np.linalg.norm(pt - intersections[j]) < 20:
                cluster.append(intersections[j])
                taken[j] = True
        clusters.append(np.mean(cluster, axis=0))
    clusters = np.array(clusters)
    if len(clusters) < 4:
        return None
    # En uzak 4 noktayı al
    center = np.mean(clusters, axis=0)
    dists = np.linalg.norm(clusters - center, axis=1)
    idxs = np.argsort(dists)[-4:]
    return clusters[idxs]


def good_features_polygon_corners(edge_img, max_corners, quality_level, min_distance):
    corners = cv2.goodFeaturesToTrack(edge_img, maxCorners=max_corners, qualityLevel=quality_level, minDistance=min_distance)
    if corners is None or len(corners) < 4:
        return None
    corners = np.squeeze(corners)
    if corners.ndim == 1:
        corners = corners[np.newaxis, :]
    hull = cv2.convexHull(corners.reshape(-1,1,2))
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    if len(approx) == 4:
        return approx.reshape(4, 2)
    # Approx 4 değilse en uzak 4 noktayı al
    center = np.mean(corners, axis=0)
    dists = np.linalg.norm(corners - center, axis=1)
    idxs = np.argsort(dists)[-4:]
    return corners[idxs]

def detect_corners(img, max_corners=10, quality=0.01, min_distance=30):
    corners = cv2.goodFeaturesToTrack(img, maxCorners=max_corners, qualityLevel=quality, minDistance=min_distance)
    if corners is None or len(corners) < 4:
        return None

    # (N,1,2) → (N,2) yap
    corners = corners.reshape(-1, 2)

    if corners.shape[0] > 4:
        center = np.mean(corners, axis=0)
        dists = np.linalg.norm(corners - center, axis=1)
        idxs = np.argsort(dists)[-4:]
        corners = corners[idxs]

    return reorder_corners(corners)


def reorder_corners(corners):
    if corners.shape[0] != 4:
        raise ValueError("Köşe sayısı 4 olmalı.")
    mean = np.mean(corners, axis=0)
    ordered = np.zeros((4, 2), dtype=corners.dtype)
    for c in corners:
        if c[0] < mean[0] and c[1] < mean[1]:
            ordered[0] = c  # üst sol
        elif c[0] > mean[0] and c[1] < mean[1]:
            ordered[1] = c  # üst sağ
        elif c[0] > mean[0] and c[1] > mean[1]:
            ordered[2] = c  # alt sağ
        else:
            ordered[3] = c  # alt sol
    return ordered


def try_adaptive_params(img, param_sets):
    best_corners = None
    best_area = 0
    best_pre = None
    best_edges = None

    for params in param_sets:
        pre = preprocess(img, **params['preprocess'])
        edges = cv2.Canny(pre, params['canny_min'], params['canny_max'])

        # Hough yöntemi
        hough_corners = hough_intersections(edges)

        # Good features + polygon approx
        good_corners = good_features_polygon_corners(edges, **params['good_features'])

        candidates = []
        if hough_corners is not None:
            candidates.append(hough_corners)
        if good_corners is not None:
            candidates.append(good_corners)

        for corners in candidates:
            if corners is None or len(corners) != 4:
                continue
            corners = reorder_corners(corners)
            # Alanı kontrol et (en büyük alanı seç)
            area = cv2.contourArea(corners.astype(np.float32))
            if area > best_area:
                best_area = area
                best_corners = corners
                best_pre = pre
                best_edges = edges

    return best_corners, best_pre, best_edges


def correct_perspective(img, param_sets, output_ratio=0.707, intermediate=False):
    corners, pre, edges = try_adaptive_params(img, param_sets)
    if corners is None:
        raise ValueError("Yeterli köşe bulunamadı")

    h, w = img.shape[:2]
    min_dim = min(h, w)
    new_w, new_h = int(min_dim), int(min_dim * output_ratio)
    dst = np.float32([[0, 0], [new_w, 0], [new_w, new_h], [0, new_h]])

    M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    warped = cv2.warpPerspective(img, M, (new_w, new_h))

    if intermediate:
        return PILImage.fromarray(pre), PILImage.fromarray(edges), PILImage.fromarray(warped)
    else:
        return (PILImage.fromarray(warped),)


class PerspectiveCorrection(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.param_sets = [
            {
                "preprocess": {"clahe_clip": 3.0, "clahe_grid": (8, 8), "gamma": 1.0, "blur_ksize": 5},
                "canny_min": 50,
                "canny_max": 150,
                "good_features": {"max_corners": 50, "quality_level": 0.01, "min_distance": 10},
            },
            {
                "preprocess": {"clahe_clip": 2.0, "clahe_grid": (4, 4), "gamma": 1.2, "blur_ksize": 3},
                "canny_min": 30,
                "canny_max": 100,
                "good_features": {"max_corners": 30, "quality_level": 0.03, "min_distance": 5},
            },
            # İstersen buraya ek parametre setleri ekleyebilirsin
        ]
        self.output_ratio = self.request.get_param("output_ratio") or 0.707
        self.intermediate = self.request.get_param("intermediate") or False
        self.image = self.request.get_param("inputImage")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def run(self):
        img = Image.get_frame(img=self.image, redis_db=self.redis_db)
        img_np = img.value

        if img_np.dtype != np.uint8:
            img_np = (img_np * 255).astype(np.uint8) if img_np.max() <= 1.0 else img_np.astype(np.uint8)

        result = correct_perspective(img_np, self.param_sets, self.output_ratio, self.intermediate)
        if self.intermediate:
            pre_img, edges_img, warped_img = result
            img.value = np.array(warped_img)
        else:
            warped_img, = result
            img.value = np.array(warped_img)

        self.image = Image.set_frame(img=img, package_uID=self.uID, redis_db=self.redis_db)
        return build_response(context=self)


if __name__ == "__main__":
    Executor(sys.argv[1]).run()
