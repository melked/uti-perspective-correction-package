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


def preprocess(img, clahe_clip=3.0, clahe_grid=(8, 8), gamma=1.0, blur_ksize=5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    enhanced = clahe.apply(gray)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    gamma_corrected = cv2.LUT(enhanced, table)
    blurred = cv2.GaussianBlur(gamma_corrected, (blur_ksize, blur_ksize), 0)
    return blurred


def detect_corners(img, max_corners=50, quality=0.01, min_distance=10):
    corners = cv2.goodFeaturesToTrack(img, maxCorners=max_corners, qualityLevel=quality, minDistance=min_distance)
    if corners is None or len(corners) < 4:
        return None
    corners = np.squeeze(corners)
    if corners.ndim == 1:
        corners = corners[np.newaxis, :]  # Tek köşe varsa 2D yap
    if len(corners) > 4:
        center = np.mean(corners, axis=0)
        dists = np.linalg.norm(corners - center, axis=1)
        idxs = np.argsort(dists)[-4:]
        corners = corners[idxs]
    return reorder_corners(corners)



def hull_approx_corners(corners):
    hull = cv2.convexHull(corners.reshape(-1, 1, 2))
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    if len(approx) == 4:
        return approx.reshape(4, 2)
    return None


def hough_line_corners(edge_img, min_line_len=50, max_line_gap=10):
    lines = cv2.HoughLinesP(edge_img, 1, np.pi / 180, threshold=80, minLineLength=min_line_len, maxLineGap=max_line_gap)
    if lines is None:
        return None

    lines = lines[:, 0, :]  # simplify shape
    points = []
    for (x1, y1, x2, y2) in lines:
        points.append((x1, y1))
        points.append((x2, y2))

    intersections = []
    for (p1, p2), (p3, p4) in combinations(points, 2):
        denom = (p1[0] - p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] - p4[0])
        if denom == 0:
            continue
        x = ((p1[0]*p2[1] - p1[1]*p2[0])*(p3[0]-p4[0]) - (p1[0]-p2[0])*(p3[0]*p4[1] - p3[1]*p4[0])) / denom
        y = ((p1[0]*p2[1] - p1[1]*p2[0])*(p3[1]-p4[1]) - (p1[1]-p2[1])*(p3[0]*p4[1] - p3[1]*p4[0])) / denom
        if 0 <= x < edge_img.shape[1] and 0 <= y < edge_img.shape[0]:
            intersections.append([x, y])

    if len(intersections) < 4:
        return None

    intersections = np.array(intersections)
    clustered = []
    taken = np.zeros(len(intersections), dtype=bool)
    for i, pt in enumerate(intersections):
        if taken[i]:
            continue
        cluster = [pt]
        taken[i] = True
        for j, other_pt in enumerate(intersections[i+1:], i+1):
            if np.linalg.norm(pt - other_pt) < 20:
                cluster.append(other_pt)
                taken[j] = True
        clustered.append(np.mean(cluster, axis=0))
    clustered = np.array(clustered)

    if len(clustered) >= 4:
        center = np.mean(clustered, axis=0)
        dists = np.linalg.norm(clustered - center, axis=1)
        idxs = np.argsort(dists)[-4:]
        corners = clustered[idxs]
        if corners.ndim == 1:
            corners = corners[np.newaxis, :]
        return corners
    return None


def reorder_corners(corners):
    mean = np.mean(corners, axis=0)
    ordered = np.zeros((4, 2), dtype=corners.dtype)
    for c in corners:
        if c[0] < mean[0] and c[1] < mean[1]:
            ordered[0] = c  # upper-left
        elif c[0] > mean[0] and c[1] < mean[1]:
            ordered[1] = c  # upper-right
        elif c[0] > mean[0] and c[1] > mean[1]:
            ordered[2] = c  # lower-right
        else:
            ordered[3] = c  # lower-left
    return ordered


def correct_perspective(img, params):
    # Adaptif ön işleme parametreleri dizisi:
    clahe_clips = [params.get("clahe_clip", 3.0), 2.0, 4.0]
    gammas = [params.get("gamma", 1.0), 0.8, 1.2]
    blur_ksizes = [params.get("blur_ksize", 5), 3, 7]

    canny_min = params.get("canny_min", 50)
    canny_max = params.get("canny_max", 150)

    h, w = img.shape[:2]
    output_ratio = params.get("output_ratio", 0.707)

    for clahe_clip in clahe_clips:
        for gamma in gammas:
            for blur_ksize in blur_ksizes:
                pre = preprocess(img, clahe_clip, (8, 8), gamma, blur_ksize)
                edges = cv2.Canny(pre, canny_min, canny_max)

                corners = detect_corners(edges, max_corners=50, quality=0.01, min_distance=10)
                if corners is not None and len(corners) >= 4:
                    hull_corners = hull_approx_corners(corners)
                    if hull_corners is not None:
                        ordered = reorder_corners(hull_corners)
                        dst = np.float32([[0, 0], [w, 0], [w, int(w * output_ratio)], [0, int(w * output_ratio)]])
                        M = cv2.getPerspectiveTransform(ordered.astype(np.float32), dst)
                        warped = cv2.warpPerspective(img, M, (w, int(w * output_ratio)))
                        return warped

                # Alternatif HoughLines ile
                hough_corners = hough_line_corners(edges)
                if hough_corners is not None:
                    ordered = reorder_corners(hough_corners)
                    dst = np.float32([[0, 0], [w, 0], [w, int(w * output_ratio)], [0, int(w * output_ratio)]])
                    M = cv2.getPerspectiveTransform(ordered.astype(np.float32), dst)
                    warped = cv2.warpPerspective(img, M, (w, int(w * output_ratio)))
                    return warped

    raise ValueError("Yeterli köşe bulunamadı")


class PerspectiveCorrection(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.params = {
            "clahe_clip": self.request.get_param("clahe_clip") or 3.0,
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
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)

        warped = correct_perspective(img_np, self.params)
        img.value = warped
        self.image = Image.set_frame(img=img, package_uID=self.uID, redis_db=self.redis_db)
        return build_response(context=self)


if __name__ == "__main__":
    Executor(sys.argv[1]).run()
