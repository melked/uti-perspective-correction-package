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


def order_points(pts):
    """
    Köşeleri sıralar: top-left, top-right, bottom-right, bottom-left
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def enhance_image(img):
    """
    Görüntüyü CLAHE, gamma correction ve unsharp mask ile güçlendirir.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE ile kontrast artırma
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Gamma correction otomatik hesaplama
    mean = np.mean(enhanced) / 255.0
    gamma = np.log(0.5) / np.log(mean) if mean > 0 else 1.0
    gamma = np.clip(gamma, 0.3, 3.0)
    table = np.array([(i / 255.0) ** (1.0 / gamma) * 255 for i in range(256)]).astype(np.uint8)
    gamma_corrected = cv2.LUT(enhanced, table)

    # Unsharp mask (keskinleştirme)
    blur = cv2.GaussianBlur(gamma_corrected, (0, 0), 3)
    sharp = cv2.addWeighted(gamma_corrected, 1.5, blur, -0.5, 0)

    return sharp


def find_line_intersections(lines, img_shape):
    """
    Hough çizgilerinin kesişim noktalarını bulur.
    """
    def line_params(rho, theta):
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        # iki nokta (x1,y1), (x2,y2)
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        return (x1, y1), (x2, y2)

    intersections = []
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            rho1, theta1 = lines[i][0]
            rho2, theta2 = lines[j][0]
            # Doğrular paralel mi kontrolü
            if abs(theta1 - theta2) < np.deg2rad(10):
                continue

            (x1, y1), (x2, y2) = line_params(rho1, theta1)
            (x3, y3), (x4, y4) = line_params(rho2, theta2)

            # İki doğrunun kesişim noktası (Cramer yöntemi)
            denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
            if denom == 0:
                continue
            px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
            py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom

            # Görüntü sınırları içinde mi?
            if 0 <= px < img_shape[1] and 0 <= py < img_shape[0]:
                intersections.append([px, py])

    return np.array(intersections, dtype=np.float32)


def filter_corners(corners):
    """
    Aynı noktaya yakın çoklu köşeleri grupla ve merkezlerini al.
    """
    if len(corners) == 0:
        return np.array([])

    # Kümeleme için epsilon
    epsilon = 20.0
    grouped = []

    for pt in corners:
        added = False
        for group in grouped:
            if np.linalg.norm(np.array(group) - pt) < epsilon:
                # Güncelle ortalama
                group[0] = (group[0] + pt[0]) / 2
                group[1] = (group[1] + pt[1]) / 2
                added = True
                break
        if not added:
            grouped.append([pt[0], pt[1]])

    return np.array(grouped, dtype=np.float32)


def detect_corners_by_lines(img):
    """
    Güçlü ön işleme sonrası Canny, Hough ile çizgileri bul,
    çizgi kesişimlerini hesapla ve 4 köşe seç.
    """
    enhanced = enhance_image(img)

    edges = cv2.Canny(enhanced, 50, 150, apertureSize=3)
    edges = cv2.dilate(edges, np.ones((3,3),np.uint8), iterations=1)

    lines = cv2.HoughLines(edges, 1, np.pi/180, 150)
    if lines is None:
        return None

    intersections = find_line_intersections(lines, img.shape)
    if len(intersections) < 4:
        return None

    filtered = filter_corners(intersections)

    if len(filtered) < 4:
        return None

    # Merkeze göre en uzak 4 noktayı seç (belge köşeleri)
    center = np.mean(filtered, axis=0)
    dists = np.linalg.norm(filtered - center, axis=1)
    idxs = np.argsort(dists)[-4:]
    corners = filtered[idxs]

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

    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return warped


def correct_perspective(img):
    corners = detect_corners_by_lines(img)
    if corners is None:
        raise ValueError("Belge köşeleri tespit edilemedi.")

    warped = four_point_transform(img, corners)
    return PILImage.fromarray(warped)


class PerspectiveCorrection(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
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

        result_img = correct_perspective(img_np)
        img.value = np.array(result_img)
        self.image = Image.set_frame(img=img, package_uID=self.uID, redis_db=self.redis_db)
        return build_response(context=self)


if __name__ == "__main__":
    Executor(sys.argv[1]).run()
