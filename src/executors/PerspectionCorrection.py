import os
import sys
import cv2
import numpy as np
from PIL import Image as PILImage

# Kullanıcının istediği gibi sys.path eklemesini koruyoruz
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveCorrection.src.utils.response import build_response
from components.PerspectiveCorrection.src.models.PackageModel import PackageModel


def reorder_corners(corners: np.ndarray) -> np.ndarray:
    """Köşeleri şu sıraya getirir: üst-sol, üst-sağ, alt-sağ, alt-sol.
    corners: (4,2) or (N,2) array
    """
    pts = corners.reshape((4, 2))
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(4)
    ordered = np.zeros((4, 2), dtype=pts.dtype)
    ordered[0] = pts[np.argmin(s)]  # top-left
    ordered[2] = pts[np.argmax(s)]  # bottom-right
    ordered[1] = pts[np.argmin(diff)]  # top-right
    ordered[3] = pts[np.argmax(diff)]  # bottom-left
    return ordered


def preprocess(img: np.ndarray,
               clahe_clip: float = 3.0,
               clahe_grid: tuple = (8, 8),
               gamma: float = 1.0,
               blur_ksize: int = 5) -> np.ndarray:
    """Temel kontrast/gamma/denge ön işlemleri yapar. Gri dönüşüm döner.
    - img: BGR image (uint8)
    - dönen sonuç grayscale uint8
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    clahe = cv2.createCLAHE(clipLimit=max(0.1, clahe_clip), tileGridSize=clahe_grid)
    enhanced = clahe.apply(gray)

    if gamma != 1.0 and gamma > 0:
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        enhanced = cv2.LUT(enhanced, table)

    if blur_ksize and blur_ksize > 1:
        k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        enhanced = cv2.GaussianBlur(enhanced, (k, k), 0)

    return enhanced


def find_largest_quadrilateral(thresh: np.ndarray, min_area: float = 1000) -> np.ndarray:
    """Binary (0/255) görüntüden en uygun dörtgen konturu bulmaya çalışır.
    Eğer bulamazsa None döner.
    """
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            # Shape regularity: daha düzgün kenar uzunluklarına öncelik ver
            pts = approx.reshape(4, 2)
            lens = [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]
            side_ratio = max(lens) / (min(lens) + 1e-6)
            ratio_score = 1.0 / (1.0 + abs(side_ratio - 1.414))
            score = area * ratio_score
            if score > best_score:
                best_score = score
                best = approx.reshape(4, 2)

    return best


def detect_corners_combined(img: np.ndarray, debug: bool = False) -> np.ndarray:
    """Belge köşelerini tespit etmek için birkaç yöntemi birleştirir.
    - img: grayscale image (uint8)
    """
    h, w = img.shape[:2]
    min_area = h * w * 0.02  # daha esnek; belge daha küçük olabilir

    candidates = []

    # 1) Adaptive threshold + morphology -> kontur tabanlı en iyi dörtgen
    adapt = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(adapt, cv2.MORPH_CLOSE, kernel, iterations=2)
    quad = find_largest_quadrilateral(closed, min_area)
    if quad is not None:
        candidates.append(quad)

    # 2) Canny + kontur -> başka bir kontur kaynağı
    edges = cv2.Canny(img, 50, 150)
    closed_e = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    quad2 = find_largest_quadrilateral(closed_e, min_area)
    if quad2 is not None:
        candidates.append(quad2)

    # 3) Hough Lines kesişimleri (düz ve dik çizgiler için)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, int(min(h, w) / 3))
    if lines is not None:
        horiz = []
        vert = []
        for l in lines[:, 0]:
            rho, theta = l
            angle = np.degrees(theta) % 180
            if angle < 10 or angle > 170:
                horiz.append((rho, theta))
            elif 80 < angle < 100:
                vert.append((rho, theta))
        if len(horiz) >= 2 and len(vert) >= 2:
            intersections = []
            for (r1, t1) in horiz[:3]:
                for (r2, t2) in vert[:3]:
                    cos1, sin1 = np.cos(t1), np.sin(t1)
                    cos2, sin2 = np.cos(t2), np.sin(t2)
                    denom = cos1 * sin2 - sin1 * cos2
                    if abs(denom) < 1e-6:
                        continue
                    x = (sin2 * r1 - sin1 * r2) / denom
                    y = (cos1 * r2 - cos2 * r1) / denom
                    if 0 <= x < w and 0 <= y < h:
                        intersections.append([x, y])
            if len(intersections) >= 4:
                intersections = np.array(intersections)
                c = np.mean(intersections, axis=0)
                dists = np.linalg.norm(intersections - c, axis=1)
                cand = intersections[np.argsort(dists)[-4:]]
                candidates.append(cand)

    # 4) goodFeaturesToTrack (köşe yoğunluğu olan belgeler için)
    corners = cv2.goodFeaturesToTrack(img, maxCorners=32, qualityLevel=0.01, minDistance=min(h, w) // 50)
    if corners is not None and len(corners) >= 4:
        corners = np.squeeze(corners)
        # en dıştaki 4 noktayı al (belge köşelerine yakın olacak)
        center = np.mean(corners, axis=0)
        dists = np.linalg.norm(corners - center, axis=1)
        cand = corners[np.argsort(dists)[-4:]]
        candidates.append(cand)

    # Adayları değerlendirme
    best = None
    best_score = -1
    for cand in candidates:
        if cand is None or len(cand) < 4:
            continue
        cand = reorder_corners(np.array(cand, dtype=np.float32))
        area = cv2.contourArea(cand.astype(np.float32))
        if area < min_area:
            continue
        # kenar uzunlukları ve orantıya bak
        side_lens = [np.linalg.norm(cand[i] - cand[(i + 1) % 4]) for i in range(4)]
        ratio = max(side_lens) / (min(side_lens) + 1e-6)
        a4_score = 1 / (1 + abs(ratio - 1.414))
        score = area * a4_score
        if score > best_score:
            best_score = score
            best = cand

    # Fallback: image corners (küçük margin ile)
    if best is None:
        margin = max(10, int(min(h, w) * 0.03))
        best = np.array([
            [margin, margin],
            [w - margin, margin],
            [w - margin, h - margin],
            [margin, h - margin]
        ], dtype=np.float32)

    # Rötuş: subpixel doğruluğu artırmak için köşeleri iyileştir
    try:
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
        best_sub = best.astype(np.float32)
        # cornerSubPix requires single-channel float32 image
        gray_f = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.cornerSubPix(gray_f, best_sub, (5, 5), (-1, -1), term)
        best = best_sub
    except Exception:
        pass

    return best


def correct_perspective(img: np.ndarray, params: dict) -> PILImage.Image:
    """Perspektif düzeltme ana fonksiyonu.
    - img: BGR uint8
    - params: dict ile ayarlar
    Döndürülmüş PIL Image döner
    """
    pre = preprocess(img, params.get("clahe_clip", 3.0), params.get("clahe_grid", (8, 8)),
                     params.get("gamma", 1.0), params.get("blur_ksize", 5))

    # Kenarları al
    canny_min = params.get("canny_min", 50)
    canny_max = params.get("canny_max", 150)
    edges = cv2.Canny(pre, canny_min, canny_max)

    corners = detect_corners_combined(pre)
    if corners is None or len(corners) < 4:
        raise ValueError("Yeterli köşe bulunamadı")

    h, w = img.shape[:2]
    # hedef boyut: belge oranına göre esnek
    # output_ratio default olarak 0.707 (A4 kare kök 2 oranının tersine göre) gelebilir
    output_ratio = params.get("output_ratio", 0.707)
    min_dim = min(h, w)
    new_w = int(min_dim)
    new_h = int(min_dim * output_ratio)

    dst = np.float32([[0, 0], [new_w - 1, 0], [new_w - 1, new_h - 1], [0, new_h - 1]])
    src = corners.astype(np.float32)
    src = reorder_corners(src)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR)

    # Opsiyonel: kenar kırpmayı otomatik yap (beyaz kenarlar)
    gray_w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray_w, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # kenar boşluklarını bul
    coords = cv2.findNonZero(255 - th)
    if coords is not None:
        x, y, ww, hh = cv2.boundingRect(coords)
        cropped = warped[y:y + hh, x:x + ww]
    else:
        cropped = warped

    return PILImage.fromarray(cropped)


class PerspectiveCorrection(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        # varsayılanları burada açıkça set et (dışarıdan override edilebilir)
        self.params = {
            "clahe_clip": float(self.request.get_param("clahe_clip") or 3.0),
            "clahe_grid": tuple(self.request.get_param("clahe_grid") or (8, 8)),
            "gamma": float(self.request.get_param("gamma") or 1.0),
            "blur_ksize": int(self.request.get_param("blur_ksize") or 5),
            "canny_min": int(self.request.get_param("canny_min") or 50),
            "canny_max": int(self.request.get_param("canny_max") or 150),
            "output_ratio": float(self.request.get_param("output_ratio") or 0.707),
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

        # Güvenlik: çok küçük resimlerde hata verme
        if min(img_np.shape[:2]) < 50:
            raise ValueError("Girdi görüntüsü çok küçük")

        result = correct_perspective(img_np, self.params)

        img.value = np.array(result)
        self.image = Image.set_frame(img=img, package_uID=self.uID, redis_db=self.redis_db)
        return build_response(context=self)


if __name__ == "__main__":
    Executor(sys.argv[1]).run()
