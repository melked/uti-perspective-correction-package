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
    intersections = [[[] for _ in lines] for _ in lines]

    for i, (x1, y1, x2, y2) in enumerate(lines):
        for j, (x3, y3, x4, y4) in enumerate(lines):
            if i >= j:
                continue  # Simetrik olduğundan işlem yapma

            d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if d == 0:
                continue  # Paralel çizgiler

            x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d
            y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d

            if 0 <= x <= width and 0 <= y <= height:
                intersections[i][j] = (int(x), int(y))
                intersections[j][i] = (int(x), int(y))

    return intersections


def annotate_intersections(img, intersections):
    temp = np.array(img)
    for row in intersections:
        for point in row:
            if point:
                cv2.circle(temp, point, 20, RED, 10)
    return temp


def annotate_corners(img, corners):
    temp = np.array(img)
    for x, y in corners:
        cv2.circle(temp, (int(x), int(y)), 20, GREEN, 10)
    return temp


def get_corners(intersections):
    # Tüm kesişim noktalarını listele, 4 tanesini al
    pts = list(set(pt for row in intersections for pt in row if pt))
    return np.array(pts, dtype=np.float32)[:4]


def filter_perpendicular(lines, margin):
    perpendicular = np.pi / 2
    count = np.zeros(len(lines))

    from itertools import combinations
    for a, b in combinations(range(len(lines)), 2):
        angle_diff = abs(lines[a][1] - lines[b][1])
        if abs(angle_diff - perpendicular) < margin:
            count[a] += 1
            count[b] += 1

    lines = np.array(lines)
    return lines[count >= 2]


def line_distance(line_a, line_b):
    rho_a, theta_a = line_a
    rho_b, theta_b = line_b
    dist = rho_a ** 2 + rho_b ** 2 - 2 * rho_a * rho_b * np.cos(theta_a - theta_b)
    return np.sqrt(dist)


def eliminate_duplicates(img, lines, threshold):
    eliminated = np.zeros(len(lines), dtype=bool)
    min_dist = max(img.shape[:2]) * threshold
    min_theta = np.pi * threshold

    from itertools import combinations
    for i, j in combinations(range(len(lines)), 2):
        if eliminated[i] or eliminated[j]:
            continue

        theta_diff = abs(lines[i][1] - lines[j][1])
        if theta_diff > np.pi / 2:
            theta_diff = np.pi - theta_diff

        if line_distance(lines[i], lines[j]) < min_dist and theta_diff < min_theta:
            eliminated[i] = True

    return lines[~eliminated]


def to_cartesian(img, lines):
    height, width, _ = img.shape
    coeff = max(height, width)
    cartesian = []
    for rho, theta in lines:
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        x1, y1 = int(x0 + coeff * (-b)), int(y0 + coeff * a)
        x2, y2 = int(x0 - coeff * (-b)), int(y0 - coeff * a)
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
        x, y = corner
        if x < mean[0] and y < mean[1]:
            new_corners[0] = corner  # üst sol
        elif x > mean[0] and y < mean[1]:
            new_corners[1] = corner  # üst sağ
        elif x > mean[0] and y > mean[1]:
            new_corners[2] = corner  # alt sağ
        else:
            new_corners[3] = corner  # alt sol

    return new_corners


def correct_perspective(
    img,
    threshold_max=140,
    threshold_min=30,
    median_blur_size=51,
    rho=1,
    theta=np.pi / 180,
    threshold_intersect=250,
    threshold_distance=0.15,
    perpendicular_margin=np.pi / 18,
    intermediate=True,
):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if intermediate:
        gray_im = PILImage.fromarray(gray)

    blurred = cv2.medianBlur(gray, median_blur_size)
    if intermediate:
        blurred_im = PILImage.fromarray(blurred)

    edges = cv2.Canny(blurred, threshold_min, threshold_max)
    if intermediate:
        edges_im = PILImage.fromarray(edges)

    lines = cv2.HoughLines(edges, rho, theta, threshold_intersect)
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
    corners = get_corners(intersections)
    print("number of corners:", len(corners))
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
        self.request.model = PackageModel(**self.request.data)
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
            img_np = (img_np * 255).astype(np.uint8) if img_np.max() <= 1.0 else img_np.astype(np.uint8)

        corrected_tuple = correct_perspective(img_np, intermediate=False)
        corrected_img = corrected_tuple[0]

        img.value = np.array(corrected_img)

        self.image = Image.set_frame(img=img, package_uID=self.uID, redis_db=self.redis_db)

        return build_response(context=self)


if __name__ == "__main__":
    Executor(sys.argv[1]).run()
