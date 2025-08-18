import os
import sys
import cv2
import numpy as np
from typing import Optional

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveCorrection.src.utils.response import build_response
from components.PerspectiveCorrection.src.models.PackageModel import PackageModel


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
    if maxWidth == 0 or maxHeight == 0: return None
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LANCZOS4)


def stage1_fast_and_simple(image: np.ndarray) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > (image.shape[0] * image.shape[1] * 0.2):
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                return approx.reshape(4, 2).astype(np.float32)
    return None

def stage2_smart_and_scored(image: np.ndarray) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(gray, 11, 75, 75)
    thresh = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 7)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 10))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=5)
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    h, w = image.shape[:2];
    img_center = np.array([w / 2, h / 2])
    best_score = -1;
    best_quad = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad = approx.reshape(4, 2).astype(np.float32)
            area = cv2.contourArea(quad);
            area_score = np.clip(area / (w * h), 0, 1)
            M = cv2.moments(quad);
            if M["m00"] == 0: continue
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            dist = np.linalg.norm(np.array([cx, cy]) - img_center)
            centrality_score = 1 - np.clip(dist / (max(w, h) / 2), 0, 1)
            final_score = (area_score * 0.6) + (centrality_score * 0.4)
            if final_score > best_score:
                best_score, best_quad = final_score, quad
    return best_quad

def stage3_color_clustering(image: np.ndarray) -> Optional[np.ndarray]:
    h, w = image.shape[:2]
    scale = 400 / max(h, w)
    small_img = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    pixels = small_img.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = centers.astype(np.uint8)
    lab_centers = cv2.cvtColor(centers.reshape(1, -1, 3), cv2.COLOR_BGR2LAB)[0]
    luminance = [c[0] for c in lab_centers]
    counts = np.bincount(labels.flatten())
    best_cluster_idx = -1;
    max_score = -1
    for i in range(len(centers)):
        if luminance[i] < 60: continue
        score = luminance[i] * counts[i]
        if score > max_score:
            max_score, best_cluster_idx = score, i
    if best_cluster_idx == -1: best_cluster_idx = np.argmax(counts)
    mask = (labels.reshape(small_img.shape[:2]) == best_cluster_idx).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < w * h * 0.1: return None
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    return box.astype(np.float32)

class PerspectiveCorrection(Component):
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

        document_quad = None


        print("Aşama 1 (Hızlı Gözcü) deneniyor...")
        document_quad = stage1_fast_and_simple(src_img)

        # Aşama 2: Uzman Doktor (Akıllı Yargıç)
        if document_quad is None:
            print("Aşama 1 başarısız. Aşama 2 (Akıllı Yargıç) deneniyor...")
            document_quad = stage2_smart_and_scored(src_img)

        # Aşama 3: Konsültan Profesör (Renk Avcısı)
        if document_quad is None:
            print("Aşama 2 başarısız. Aşama 3 (Renk Avcısı) deneniyor...")
            document_quad = stage3_color_clustering(src_img)

        if document_quad is not None:
            print("Başarılı: Uygun bir belge adayı bulundu.")
        else:
            print("Tüm uzmanlar başarısız. Fallback olarak tüm görüntü kullanılıyor.")
            document_quad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

        warped = _four_point_transform(src_img, document_quad)
        if warped is None:
            print("Dönüşüm hatası, fallback kullanılıyor.")
            warped = src_img
            document_quad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)
        self.context["src_quad"] = document_quad.tolist()
        self.context["output_size"] = [warped.shape[1], warped.shape[0]]
        return build_response(context=self)

Executor(sys.argv[1]).run()