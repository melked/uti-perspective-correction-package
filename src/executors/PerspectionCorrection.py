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


# ---------- UTILS ----------
def order_points(pts):
    """Köşeleri (top-left, top-right, bottom-right, bottom-left) sırasına dizer."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]     # top-left
    rect[2] = pts[np.argmax(s)]     # bottom-right
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def preprocess(img, clahe_clip=3.0, clahe_grid=(8, 8), gamma=1.0, blur_ksize=5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    enhanced = clahe.apply(gray)

    # Gamma correction
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    gamma_corrected = cv2.LUT(enhanced, table)

    blurred = cv2.GaussianBlur(gamma_corrected, (blur_ksize, blur_ksize), 0)
    return blurred


def detect_corners_gftt(img, max_corners=10, quality=0.01, min_distance=30):
    """goodFeaturesToTrack ile köşe tespiti."""
    corners = cv2.goodFeaturesToTrack(img, maxCorners=max_corners, qualityLevel=quality, minDistance=min_distance)
    if corners is None or len(corners) < 4:
        return None
    corners = np.squeeze(corners)
    if len(corners) > 4:
        center = np.mean(corners, axis=0)
        dists = np.linalg.norm(corners - center, axis=1)
        idxs = np.argsort(dists)[-4:]
        corners = corners[idxs]
    return order_points(corners)


def detect_corners_contour(img):
    """Kontur ile en büyük dörtgen köşelerini bulma (fallback)."""
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            return order_points(np.squeeze(approx))
    return None


def compute_dynamic_ratio(corners):
    """Köşe mesafelerinden dinamik en-boy oranı hesaplar."""
    (tl, tr, br, bl) = corners
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))
    return maxWidth, maxHeight


# ---------- MAIN PROCESS ----------
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
    output_ratio=None,
    intermediate=False,
):
    pre = preprocess(img, clahe_clip, clahe_grid, gamma, blur_ksize)

    # Gürültü azaltma - morfolojik işlemler
    edges = cv2.Canny(pre, canny_min, canny_max)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Köşe tespiti (önce GFTT, olmazsa kontur)
    corners = detect_corners_gftt(edges, max_corners, quality_level, min_distance)
    if corners is None:
        corners = detect_corners_contour(edges)

    if corners is None:
        raise ValueError("Köşeler tespit edilemedi")

    # Dinamik boyut hesaplama
    if output_ratio is None:
        new_w, new_h = compute_dynamic_ratio(corners)
    else:
        h, w = img.shape[:2]
        min_dim = min(h, w)
        new_w, new_h = int(min_dim), int(min_dim * output_ratio)

    dst = np.float32([[0, 0], [new_w, 0], [new_w, new_h], [0, new_h]])
    M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    warped = cv2.warpPerspective(img, M, (new_w, new_h))

    if intermediate:
        return (
            PILImage.fromarray(pre),
            PILImage.fromarray(edges),
            PILImage.fromarray(warped),
        )
    else:
        return (PILImage.fromarray(warped),)


# ---------- COMPONENT ----------
class PerspectiveCorrection(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.params = {
            "clahe_clip": self._get_param("clahe_clip", 3.0),
            "clahe_grid": self._get_param("clahe_grid", (8, 8)),
            "gamma": self._get_param("gamma", 1.0),
            "blur_ksize": self._get_param("blur_ksize", 5),
            "canny_min": self._get_param("canny_min", 50),
            "canny_max": self._get_param("canny_max", 150),
            "max_corners": self._get_param("max_corners", 10),
            "quality_level": self._get_param("quality_level", 0.01),
            "min_distance": self._get_param("min_distance", 30),
            "output_ratio": self._get_param("output_ratio", None),  # None olursa dinamik
            "intermediate": self._get_param("intermediate", False),
        }
        self.image = self.request.get_param("inputImage")

    def _get_param(self, name, default):
        val = self.request.get_param(name)
        return default if val is None else val

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
            **self.params
        )

        img.value = np.array(result[-1])
        self.image = Image.set_frame(img=img, package_uID=self.uID, redis_db=self.redis_db)
        return build_response(context=self)


if __name__ == "__main__":
    Executor(sys.argv[1]).run()
