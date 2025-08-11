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
    """Köşeleri saat yönünde sıralar: üst-sol, üst-sağ, alt-sağ, alt-sol"""
    if corners is None or len(corners) != 4:
        return None

    # Merkez nokta
    center = np.mean(corners, axis=0)

    # Her köşenin merkeze göre açısını hesapla
    angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])

    # Açılara göre sırala (saat yönünde)
    sorted_indices = np.argsort(angles)

    # Köşeleri doğru sırada düzenle
    ordered = np.zeros((4, 2), dtype=corners.dtype)

    # En üstteki iki noktayı bul
    top_indices = sorted_indices[angles[sorted_indices] < 0]
    bottom_indices = sorted_indices[angles[sorted_indices] >= 0]

    if len(top_indices) >= 2:
        # Üst iki nokta: sol ve sağ
        top_points = corners[top_indices]
        left_idx = top_indices[np.argmin(top_points[:, 0])]
        right_idx = top_indices[np.argmax(top_points[:, 0])]
        ordered[0] = corners[left_idx]  # üst-sol
        ordered[1] = corners[right_idx]  # üst-sağ

    if len(bottom_indices) >= 2:
        # Alt iki nokta: sol ve sağ
        bottom_points = corners[bottom_indices]
        right_idx = bottom_indices[np.argmax(bottom_points[:, 0])]
        left_idx = bottom_indices[np.argmin(bottom_points[:, 0])]
        ordered[2] = corners[right_idx]  # alt-sağ
        ordered[3] = corners[left_idx]  # alt-sol

    return ordered


def preprocess(img, clahe_clip=3.0, clahe_grid=(8, 8), gamma=1.0, blur_ksize=5):
    """Görüntüyü köşe algılama için ön işleme"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE uygula
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    enhanced = clahe.apply(gray)

    # Gamma düzeltmesi
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    gamma_corrected = cv2.LUT(enhanced, table)

    # Gaussian blur
    blurred = cv2.GaussianBlur(gamma_corrected, (blur_ksize, blur_ksize), 0)
    return blurred


def detect_corners_hough_lines(img, rho=1, theta=np.pi / 180, threshold=100, min_line_length=100, max_line_gap=10):
    """Hough çizgileri kullanarak köşe algılama"""
    edges = cv2.Canny(img, 50, 150, apertureSize=3)

    # Hough çizgi algılama
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)

    if lines is None or len(lines) < 4:
        return None

    # Çizgileri uzatarak kesişim noktalarını bul
    intersections = []
    lines = lines.reshape(-1, 4)

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            # İki çizginin kesişim noktasını hesapla
            x1, y1, x2, y2 = lines[i]
            x3, y3, x4, y4 = lines[j]

            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) > 1e-6:
                t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
                x = x1 + t * (x2 - x1)
                y = y1 + t * (y2 - y1)

                # Görüntü sınırları içindeki kesişimleri al
                if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                    intersections.append([x, y])

    if len(intersections) < 4:
        return None

    intersections = np.array(intersections)

    # En uygun 4 köşeyi seç (convex hull kullanarak)
    if len(intersections) > 4:
        hull = cv2.convexHull(intersections.astype(np.float32))
        hull = hull.reshape(-1, 2)

        if len(hull) >= 4:
            # En dış 4 köşeyi seç
            center = np.mean(hull, axis=0)
            distances = np.linalg.norm(hull - center, axis=1)
            furthest_indices = np.argsort(distances)[-4:]
            intersections = hull[furthest_indices]

    return intersections[:4]


def detect_corners_contours(img):
    """Kontur analizi kullanarak köşe algılama"""
    edges = cv2.Canny(img, 50, 150)

    # Morfolojik işlemler
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Konturları bul
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # En büyük konturu seç
    largest_contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(largest_contour) < 1000:  # Çok küçük konturları filtrele
        return None

    # Convex hull uygula
    hull = cv2.convexHull(largest_contour)

    # Poligon yaklaşımı
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)

    if len(approx) >= 4:
        corners = approx.reshape(-1, 2)

        # En uygun 4 köşeyi seç
        if len(corners) > 4:
            center = np.mean(corners, axis=0)
            distances = np.linalg.norm(corners - center, axis=1)
            furthest_indices = np.argsort(distances)[-4:]
            corners = corners[furthest_indices]

        return corners[:4]

    return None


def detect_corners_goodfeatures(img, max_corners=10, quality=0.01, min_distance=30):
    """GoodFeaturesToTrack kullanarak köşe algılama (orijinal yöntem)"""
    corners = cv2.goodFeaturesToTrack(img, maxCorners=max_corners, qualityLevel=quality, minDistance=min_distance)

    if corners is None or len(corners) < 4:
        return None

    corners = np.squeeze(corners)

    if len(corners) > 4:
        center = np.mean(corners, axis=0)
        distances = np.linalg.norm(corners - center, axis=1)
        furthest_indices = np.argsort(distances)[-4:]
        corners = corners[furthest_indices]

    return corners[:4]


def evaluate_corner_quality(corners, img_shape):
    """Köşe kalitesini değerlendir"""
    if corners is None or len(corners) != 4:
        return 0

    h, w = img_shape[:2]

    # Köşelerin görüntü sınırlarına olan uzaklığını hesapla
    distances_to_border = []
    for corner in corners:
        x, y = corner
        dist = min(x, y, w - x, h - y)
        distances_to_border.append(dist)

    # Alan hesapla
    area = cv2.contourArea(corners.astype(np.float32))

    # Köşeler arası mesafelerin tutarlılığını kontrol et
    side_lengths = []
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        length = np.linalg.norm(p2 - p1)
        side_lengths.append(length)

    side_variation = np.std(side_lengths) / np.mean(side_lengths) if np.mean(side_lengths) > 0 else float('inf')

    # Skor hesapla (yüksek alan, düşük kenar varyasyonu, sınırdan uzak köşeler)
    score = area / (w * h) - side_variation * 0.1 - (1.0 / (1.0 + min(distances_to_border)))

    return max(0, score)


def detect_corners_combined(img, **params):
    """Birden fazla yöntemi birleştirerek en iyi köşeleri bul"""
    methods = [
        ("goodfeatures", lambda: detect_corners_goodfeatures(
            img,
            params.get("max_corners", 10),
            params.get("quality_level", 0.01),
            params.get("min_distance", 30)
        )),
        ("hough", lambda: detect_corners_hough_lines(img)),
        ("contours", lambda: detect_corners_contours(img))
    ]

    best_corners = None
    best_score = -1
    best_method = None

    for method_name, detect_func in methods:
        try:
            corners = detect_func()
            if corners is not None:
                ordered_corners = reorder_corners(corners)
                if ordered_corners is not None:
                    score = evaluate_corner_quality(ordered_corners, img.shape)
                    if score > best_score:
                        best_score = score
                        best_corners = ordered_corners
                        best_method = method_name
        except Exception as e:
            continue

    if best_corners is not None:
        print(f"En iyi köşe algılama yöntemi: {best_method}, Skor: {best_score:.4f}")

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
    # Ön işleme
    pre = preprocess(img, clahe_clip, clahe_grid, gamma, blur_ksize)
    edges = cv2.Canny(pre, canny_min, canny_max)

    # Birleşik köşe algılama
    corners = detect_corners_combined(
        edges,
        max_corners=max_corners,
        quality_level=quality_level,
        min_distance=min_distance
    )

    if corners is None:
        raise ValueError("Yeterli köşe bulunamadı - hiçbir yöntem başarılı olamadı")

    # Perspektif düzeltme
    h, w = img.shape[:2]
    min_dim = min(h, w)
    new_w, new_h = int(min_dim), int(min_dim * output_ratio)

    dst = np.float32([[0, 0], [new_w, 0], [new_w, new_h], [0, new_h]])
    M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    warped = cv2.warpPerspective(img, M, (new_w, new_h))

    if intermediate:
        # Debug için köşeleri işaretlenmiş görüntü oluştur
        debug_img = img.copy()
        for i, corner in enumerate(corners):
            cv2.circle(debug_img, tuple(corner.astype(int)), 10, (0, 255, 0), -1)
            cv2.putText(debug_img, str(i), tuple(corner.astype(int) + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        return (
            PILImage.fromarray(pre),
            PILImage.fromarray(edges),
            PILImage.fromarray(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)),
            PILImage.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        )
    else:
        return (PILImage.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)),)


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