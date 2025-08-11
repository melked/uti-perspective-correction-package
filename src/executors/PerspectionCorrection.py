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


def annotate_intersections(img, intersections):
    temp = np.array(img)
    for row in intersections:
        for col in row:
            if col:  # col boş değilse
                cv2.circle(temp, col, 20, RED, 10)
    return temp


def annotate_corners(img, corners):
    temp = np.array(img)
    for x, y in corners:
        cv2.circle(temp, (int(x), int(y)), 20, GREEN, 10)
    return temp


def get_corners(lines, intersections):
    # Only handle 4 lines case
    pts = list(set(i for j in intersections for i in j if i))
    return np.array(pts, dtype=np.float32)[:4]


def filter_perpendicular(lines, margin):
    perpendicular = np.pi / 2
    count = np.zeros(len(lines))

    from itertools import combinations
    idxs = combinations(range(len(lines)), 2)

    for a, b in idxs:
        if abs(abs(lines[a][1] - lines[b][1]) - perpendicular) < margin:
            count[a] += 1
            count[b] += 1

    lines = np.array(lines)
    return lines[count >= 2]


def line_distance(line_a, line_b):
    rho_a, theta_a = line_a
    rho_b, theta_b = line_b

    result = rho_a ** 2 + rho_b ** 2
    result -= 2 * rho_b * rho_a * np.cos(theta_a - theta_b)
    return np.sqrt(result)


def eliminate_duplicates(img, lines, threshold):
    eliminated = np.zeros(len(lines), dtype=bool)
    min_distance = max(img.shape[:2]) * threshold
    min_theta = np.pi * threshold

    from itertools import combinations
    idxs = combinations(range(len(lines)), 2)

    for i, j in idxs:
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


def annotate_lines(img, lines):
    annotated = np.array(img)
    for x1, y1, x2, y2 in lines:
        cv2.line(annotated, (x1, y1), (x2, y2), BLUE, 10)
    return annotated.astype(np.uint8)


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


def correct_perspective(
    img,
    threshold_max=140,
    threshold_min=30,
    gaussian_blur_size=7,
    median_blur_size=51,
    rho=1,
    theta=np.pi / 180,
    threshold_intersect=250,
    threshold_distance=0.15,
    perpendicular_margin=np.pi / 18,
    intermediate=True,
):
    # Convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if intermediate:
        gray_im = PILImage.fromarray(gray)

    # Blur
    blurred = cv2.medianBlur(gray, median_blur_size)
    if intermediate:
        blurred_im = PILImage.fromarray(blurred)

    # Edge detection
    edges = cv2.Canny(blurred, threshold_min, threshold_max)
    if intermediate:
        edges_im = PILImage.fromarray(edges)

    # Hough lines
    lines = cv2.HoughLines(edges, rho, theta, threshold_intersect)

    # Eğer lines None ise hata vermeden devam et
    if lines is None:
        raise ValueError("No lines detected for perspective correction")

    lines = filter_perpendicular(lines[0], perpendicular_margin)
    lines = eliminate_duplicates(img, lines, threshold_distance)

    while len(lines) < 4 and threshold_intersect > 0:
        threshold_intersect -= 10
        lines = cv2.HoughLines(edges, rho, theta, threshold_intersect)
        if lines is None:
            continue
        lines = filter_perpendicular(lines[0], perpendicular_margin)
        lines = eliminate_duplicates(img, lines, threshold_distance)

    if len(lines) < 4:
        raise ValueError("Could not detect enough lines for perspective correction")

    cartesian = to_cartesian(img, lines)

    if intermediate:
        lines_annotated = PILImage.fromarray(annotate_lines(img, cartesian))

    intersections = get_intersections(img, cartesian)

    corners = get_corners(lines, intersections)
    print("number of corners: ", len(corners))
    if intermediate:
        corners_annotated = PILImage.fromarray(annotate_corners(lines_annotated, corners))

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
        return gray_im, blurred_im, edges_im, lines_annotated, corners_annotated, PILImage.fromarray(final)
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

        # Tip dönüşümü ve kanal kontrolü
        if img_np.dtype != np.uint8:
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)

        # Perspektif düzeltme
        corrected_tuple = correct_perspective(img_np, intermediate=False)
        corrected_img = corrected_tuple[0]

        img.value = np.array(corrected_img)

        self.image = Image.set_frame(img=img, package_uID=self.uID, redis_db=self.redis_db)

        packageModel = build_response(context=self)
        return packageModel


if __name__ == "__main__":
    Executor(sys.argv[1]).run()
