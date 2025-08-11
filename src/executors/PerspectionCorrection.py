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


BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)


def get_intersections(img, lines):
    height, width, _ = img.shape
    line_count = len(lines)
    intersections = [[[] for _ in range(line_count)] for _ in range(line_count)]

    for i, pointsa in enumerate(lines):
        x1, y1, x2, y2 = pointsa
        for j, pointsb in enumerate(lines):
            if intersections[i][j]:
                continue
            x3, y3, x4, y4 = pointsb

            d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if d != 0:
                x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d
                y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d
                if 0 <= x <= width and 0 <= y <= height:
                    intersections[i][j] = (int(x), int(y))
                    intersections[j][i] = (int(x), int(y))

    return intersections


def annotate_corners(img, corners):
    temp = np.array(img)
    for x, y in corners:
        cv2.circle(temp, (int(x), int(y)), 20, GREEN, 10)
    return temp


def get_corners(intersections):
    pts = list(set(i for j in intersections for i in j if i))
    return np.array(pts, dtype=np.float32)[:4]


def filter_perpendicular(lines, margin):
    perpendicular = np.pi / 2
    count = np.zeros(len(lines))
    from itertools import combinations
    for a, b in combinations(range(len(lines)), 2):
        if abs(abs(lines[a][1] - lines[b][1]) - perpendicular) < margin:
            count[a] += 1
            count[b] += 1
    lines = np.array(lines)
    return lines[count >= 2]


def line_distance(line_a, line_b):
    rho_a, theta_a = line_a
    rho_b, theta_b = line_b
    result = rho_a ** 2 + rho_b ** 2 - 2 * rho_b * rho_a * np.cos(theta_a - theta_b)
    return np.sqrt(result)


def eliminate_duplicates(img, lines, threshold):
    eliminated = np.zeros(len(lines), dtype=bool)
    min_distance = max(img.shape[:2]) * threshold
    min_theta = np.pi * threshold
    from itertools import combinations
    for i, j in combinations(range(len(lines)), 2):
        if eliminated[i] or eliminated[j]:
            continue
        line_a, line_b = lines[i], lines[j]
        theta_diff = abs(line_a[1] - line_b[1])
        if theta_diff > np.pi / 2:
            theta_diff = np.pi - theta_diff
        if line_distance(line_a, line_b) < min_distance and theta_diff < min_theta:
            eliminated[i] = True
    return lines[~eliminated]


def to_cartesian(img, lines):
    height, width, _ = img.shape
    coff = max(height, width)
    cartesian = []
    for rho, theta in lines:
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        x1, y1 = int(x0 + coff * (-b)), int(y0 + coff * (a))
        x2, y2 = int(x0 - coff * (-b)), int(y0 - coff * (a))
        cartesian.append((x1, y1, x2, y2))
    return cartesian


def reorder(corners):
    new_corners = np.zeros((4, 2), dtype=corners.dtype)
    mean = np.mean(corners, axis=0)
    for corner in corners:
        if corner[0] < mean[0] and corner[1] < mean[1]:
            new_corners[0] = corner  # upper-left
        elif corner[0] > mean[0] and corner[1] < mean[1]:
            new_corners[1] = corner  # upper-right
        elif corner[0] > mean[0] and corner[1] > mean[1]:
            new_corners[2] = corner  # lower-right
        elif corner[0] < mean[0] and corner[1] > mean[1]:
            new_corners[3] = corner  # lower-left
    return new_corners


def enhance_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(gray)

    gamma = 1.2
    lookUpTable = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
    gamma_corrected = cv2.LUT(cl1, lookUpTable)

    blurred = cv2.GaussianBlur(gamma_corrected, (5, 5), 0)
    blurred = cv2.medianBlur(blurred, 5)

    gaussian = cv2.GaussianBlur(blurred, (9, 9), 10.0)
    unsharp = cv2.addWeighted(blurred, 1.5, gaussian, -0.5, 0)

    return unsharp


def correct_perspective(
    img,
    threshold_max=150,
    threshold_min=50,
    rho=1,
    theta=np.pi / 180,
    threshold_intersect_start=150,
    threshold_intersect_min=50,
    threshold_distance=0.2,
    perpendicular_margin=np.pi / 12,
    intermediate=False,
):
    processed = enhance_image(img)
    if intermediate:
        intermediate_images = []
        intermediate_images.append(PILImage.fromarray(processed))

    edges = cv2.Canny(processed, threshold_min, threshold_max)
    if intermediate:
        intermediate_images.append(PILImage.fromarray(edges))

    threshold_intersect = threshold_intersect_start
    lines = None

    while threshold_intersect >= threshold_intersect_min:
        lines = cv2.HoughLines(edges, rho, theta, threshold_intersect)
        if lines is not None:
            lines = filter_perpendicular(lines[0], perpendicular_margin)
            lines = eliminate_duplicates(img, lines, threshold_distance)
            if len(lines) >= 4:
                break
        threshold_intersect -= 10

    if lines is None or len(lines) < 4:
        raise ValueError("Could not detect enough lines for perspective correction")

    cartesian = to_cartesian(img, lines)
    if intermediate:
        intermediate_images.append(PILImage.fromarray(annotate_corners(img, get_corners(get_intersections(img, cartesian)))))

    intersections = get_intersections(img, cartesian)
    corners = get_corners(intersections)

    height, width, _ = img.shape
    min_coff = min(height, width)
    if height > width:
        new_h, new_w = int(min_coff), int(min_coff * 0.707)
    else:
        new_h, new_w = int(min_coff * 0.707), int(min_coff)

    destination = np.float32([[0, 0], [new_w, 0], [new_w, new_h], [0, new_h]])

    corners = reorder(corners)
    trans_mat = cv2.getPerspectiveTransform(corners, destination)
    final = cv2.warpPerspective(img, trans_mat, (new_w, new_h))

    if intermediate:
        intermediate_images.append(PILImage.fromarray(final))
        return tuple(intermediate_images)
    else:
        return (PILImage.fromarray(final),)


class PerspectiveCorrection(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.rotation_degree = self.request.get_param("Degree")
        self.keep_side = self.request.get_param("KeepSide")
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

        corrected_tuple = correct_perspective(
            img_np,
            threshold_max=150,
            threshold_min=50,
            threshold_intersect_start=150,
            threshold_intersect_min=50,
            threshold_distance=0.2,
            perpendicular_margin=np.pi / 12,
            intermediate=False
        )
        corrected_img = corrected_tuple[0]

        img.value = np.array(corrected_img)

        self.image = Image.set_frame(img=img, package_uID=self.uID, redis_db=self.redis_db)

        packageModel = build_response(context=self)
        return packageModel


if __name__ == "__main__":
    Executor(sys.argv[1]).run()
