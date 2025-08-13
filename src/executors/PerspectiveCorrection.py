import os
import sys
import cv2
import numpy as np
from typing import List

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveCorrection.src.utils.response import build_response
from components.PerspectiveCorrection.src.models.PackageModel import PackageModel

# ------------------------------
# Yardımcı Fonksiyonlar
def _order_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect
    maxWidth = int(round(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))))
    maxHeight = int(round(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))))
    dst = np.array([[0, 0], [maxWidth-1, 0],
                    [maxWidth-1, maxHeight-1], [0, maxHeight-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LANCZOS4)

def _full_image_quad(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    return np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=np.float32)

def _unsharp_mask(image, ksize=(5,5), strength=1.5):
    blur = cv2.GaussianBlur(image, ksize, 0)
    return cv2.addWeighted(image, 1+strength, blur, -strength, 0)

def _gamma_correction(image: np.ndarray, gamma=1.5) -> np.ndarray:
    invGamma = 1.0/gamma
    table = np.array([(i/255.0)**invGamma*255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

# ------------------------------
# Ön İşleme Yöntemleri
def _preprocess_clahe(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def _preprocess_gamma(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray)
    gamma = 1.8 if mean<80 else (0.6 if mean>180 else 1.0)
    return _gamma_correction(image, gamma)

def _preprocess_black_top_hat(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    combined = cv2.addWeighted(blackhat,1.0,tophat,1.0,0)
    return cv2.GaussianBlur(combined,(5,5),0)

def _preprocess_color_segmentation(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0,0,180]), np.array([180,30,255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

# ------------------------------
# Hough Lines ve köşe bulma
def _compute_line_intersection(l1, l2):
    x1,y1,x2,y2 = l1
    x3,y3,x4,y4 = l2
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if denom == 0: return None
    px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4))/denom
    py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4))/denom
    if np.isinf(px) or np.isinf(py) or np.isnan(px) or np.isnan(py):
        return None
    return [px, py]

def _cluster_points_simple(points, threshold=20):
    points = np.array(points)
    if len(points)==0: return np.array([])
    clustered = []
    used = np.zeros(len(points), dtype=bool)
    for i,p in enumerate(points):
        if used[i]: continue
        group = [p]
        used[i] = True
        for j,q in enumerate(points):
            if not used[j] and np.linalg.norm(p-q)<threshold:
                group.append(q)
                used[j]=True
        clustered.append(np.mean(group,axis=0))
    return np.array(clustered)

def _compute_corners_from_lines(image: np.ndarray, edges: np.ndarray, roi_mask=None):
    if roi_mask is not None:
        edges = cv2.bitwise_and(edges, edges, mask=roi_mask)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,threshold=80,minLineLength=50,maxLineGap=10)
    if lines is None or len(lines)<4: return _full_image_quad(image)

    # Paralel/dikey çizgileri al
    filtered_lines = []
    for l in lines:
        x1,y1,x2,y2 = l[0]
        angle = np.degrees(np.arctan2(y2-y1, x2-x1)) % 180
        if abs(angle-0)<15 or abs(angle-90)<15 or abs(angle-180)<15:
            filtered_lines.append([x1,y1,x2,y2])
    if len(filtered_lines)<4: return _full_image_quad(image)

    # Kesişimler
    intersections=[]
    for i in range(len(filtered_lines)):
        for j in range(i+1,len(filtered_lines)):
            pt=_compute_line_intersection(filtered_lines[i],filtered_lines[j])
            if pt is not None:
                intersections.append(pt)
    clustered = _cluster_points_simple(intersections,threshold=20)
    if len(clustered)<4: return _full_image_quad(image)

    # En dıştaki 4 nokta
    x_min,y_min = np.min(clustered,axis=0)
    x_max,y_max = np.max(clustered,axis=0)
    return np.array([[x_min,y_min],[x_max,y_min],[x_max,y_max],[x_min,y_max]],dtype=np.float32)

# ------------------------------
# ROI oluşturma (opsiyonel, belge alanını sınırlandırır)
def _approx_roi_mask(image: np.ndarray, padding_ratio=0.05):
    h,w = image.shape[:2]
    mask = np.zeros((h,w),dtype=np.uint8)
    pad_h = int(h*padding_ratio)
    pad_w = int(w*padding_ratio)
    cv2.rectangle(mask, (pad_w,pad_h),(w-pad_w,h-pad_h), 255, -1)
    return mask

# ------------------------------
# Ensemble ile en iyi dörtgeni seç
def _detect_document(image: np.ndarray) -> np.ndarray:
    candidates=[]
    roi_mask = _approx_roi_mask(image)

    methods = [
        _preprocess_clahe,
        _preprocess_gamma,
        _preprocess_black_top_hat,
        _preprocess_color_segmentation
    ]
    for m in methods:
        pre = m(image)
        edges = cv2.Canny(pre,50,150)
        quad = _compute_corners_from_lines(image,edges,roi_mask=roi_mask)
        candidates.append((quad,cv2.contourArea(quad)))

    if not candidates:
        return _full_image_quad(image)
    best_quad = max(candidates,key=lambda x:x[1])[0]
    return best_quad

# ------------------------------
# PerspectiveCorrection Component
class PerspectiveCorrection(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.context = {}
        data_dict = getattr(self.request, "data", {}) or {}
        self.request.model = PackageModel(**data_dict)
        self.image = self.request.get_param("inputImage")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def _prepare_image(self, img: np.ndarray) -> np.ndarray:
        if img is None or img.size == 0:
            raise ValueError("Input image is empty or None.")
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0,255, cv2.NORM_MINMAX).astype(np.uint8)
        if img.ndim==2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[-1]==4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def run(self):
        img_obj = Image.get_frame(img=self.image, redis_db=self.redis_db)
        if img_obj is None or img_obj.value is None:
            raise ValueError("No input image provided or failed to load.")
        src_img = self._prepare_image(img_obj.value)
        best_quad = _detect_document(src_img)
        warped = _four_point_transform(src_img, best_quad)
        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)
        self.context["src_quad"] = best_quad.tolist()
        self.context["output_size"] = [warped.shape[1], warped.shape[0]]
        return build_response(context=self)

# ------------------------------
if __name__ == "__main__":
    Executor(sys.argv[1]).run()
