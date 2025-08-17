import os
import sys
import cv2
import numpy as np

from typing import Optional

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveTransformation.src.utils.response import build_response
from components.PerspectiveTransformation.src.models.PackageModel import PackageModel


# -----------------------------------------------------------------------------
# 1. Geometri Yardımcı Fonksiyonları
# -----------------------------------------------------------------------------
def _order_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1);
    rect[0] = pts[np.argmin(s)];
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1);
    rect[1] = pts[np.argmin(diff)];
    rect[3] = pts[np.argmax(diff)]
    return rect


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl);
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br);
    heightB = np.linalg.norm(tl - bl)
    maxWidth = max(int(widthA), int(widthB));
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LANCZOS4)


# -----------------------------------------------------------------------------
# Doku Analizi ile Belge Tespiti (Gabor Filtreleri)
# -----------------------------------------------------------------------------
def find_document_by_texture(image: np.ndarray) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gabor_kernels = []
    for theta in np.arange(0, np.pi, np.pi / 4):
        kernel = cv2.getGaborKernel(
            ksize=(31, 31), sigma=4.0, theta=theta, lambd=10.0,
            gamma=0.5, psi=0, ktype=cv2.CV_32F
        )
        gabor_kernels.append(kernel)

    accum = np.zeros_like(gray, dtype=np.float32)
    for kernel in gabor_kernels:
        filtered_img = cv2.filter2D(gray, cv2.CV_32F, kernel)
        np.maximum(accum, filtered_img, accum)

    accum = cv2.normalize(accum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, thresh = cv2.threshold(accum, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=5)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < image.shape[0] * image.shape[1] * 0.1:
        return None

    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    if len(approx) == 4 and cv2.isContourConvex(approx):
        return approx.reshape(4, 2).astype(np.float32)

    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    return box.astype(np.float32)


# -----------------------------------------------------------------------------
# Ana Bileşen (Adı "PerspectiveCorrection" olarak güncellendi)
# -----------------------------------------------------------------------------
class PerspectiveCorrection(
    Component):  # <<< DEĞİŞİKLİK: Sınıfın adı isteğiniz üzerine "PerspectiveCorrection" olarak güncellendi.
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.context = {}
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def _prepare_image(self, img: np.ndarray) -> np.ndarray:
        if img is None or img.size == 0: raise ValueError("Input image is empty or None.")
        if img.dtype != np.uint8: img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def run(self):
        img_obj = Image.get_frame(img=self.image, redis_db=self.redis_db)
        if img_obj is None or img_obj.value is None: raise ValueError("No input image provided or failed to load.")

        src_img = self._prepare_image(img_obj.value)
        h, w = src_img.shape[:2]

        print("Doku Analizi (Gabor) deneniyor...")
        document_quad = find_document_by_texture(src_img)

        if document_quad is None:
            print("Tüm klasik yöntemler tükendi. Fallback olarak tüm görüntü kullanılıyor.")
            document_quad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

        warped = _four_point_transform(src_img, document_quad)
        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)
        self.context["src_quad"] = document_quad.tolist()
        self.context["output_size"] = [warped.shape[1], warped.shape[0]]
        return build_response(context=self)


# -----------------------------------------------------------------------------
# Çalıştırıcı
# -----------------------------------------------------------------------------
Executor(sys.argv[1]).run()