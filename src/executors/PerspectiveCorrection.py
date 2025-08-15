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

# ---------------- Utility Functions ---------------- #

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
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(round(max(widthA, widthB)))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(round(max(heightA, heightB)))
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0,maxHeight-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LANCZOS4)
    return warped

def _full_image_quad(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    return np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=np.float32)

def _find_quad_from_contours(binary_img: np.ndarray, ref_image: np.ndarray, min_area_ratio=0.05, max_area_ratio=0.95) -> np.ndarray:
    # ✅ Her zaman 8-bit tek kanal garantisi
    if binary_img is None or binary_img.size == 0:
        return _full_image_quad(ref_image)
    if len(binary_img.shape) == 3:  # renkli ise griye çevir
        binary_img = cv2.cvtColor(binary_img, cv2.COLOR_BGR2GRAY)
    if binary_img.dtype != np.uint8:  # 8-bit değilse normalize et
        binary_img = cv2.normalize(binary_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    contours, _ = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _full_image_quad(ref_image)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    img_area = ref_image.shape[0] * ref_image.shape[1]
    min_area = img_area * min_area_ratio
    max_area = img_area * max_area_ratio

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype(np.float32)

    return _full_image_quad(ref_image)


def _unsharp_mask(image, ksize=(5,5), strength=1.5):
    blur = cv2.GaussianBlur(image, ksize, 0)
    return cv2.addWeighted(image, 1 + strength, blur, -strength, 0)

def _gamma_correction(image: np.ndarray, gamma=1.5) -> np.ndarray:
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

# ---------------- Preprocessing & Enhancement ---------------- #

def _adaptive_contrast_enhancement(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # Ortalama parlaklık (kontrast ayarı için)
    mean_lum = np.mean(L)
    clahe = cv2.createCLAHE(
        clipLimit=2.0 if mean_lum < 100 else 3.0,
        tileGridSize=(8, 8)
    )
    cl = clahe.apply(L)
    img_clahe = cv2.cvtColor(cv2.merge((cl, A, B)), cv2.COLOR_LAB2BGR)

    # Netlik ölçümü (bulanıklık algılama)
    gray = cv2.cvtColor(img_clahe, cv2.COLOR_BGR2GRAY)
    focus_measure = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Dinamik agresif/soft ayarı
    if focus_measure < 100:  # çok bulanık
        strength = 2.0
    elif focus_measure < 300:  # hafif bulanık
        strength = 1.5
    else:  # zaten net
        strength = 1.2

    sharpened = _unsharp_mask(img_clahe, ksize=(5, 5), strength=strength)

    # Gamma düzeltmesi (çok karanlık veya çok parlak için)
    mean_gray = np.mean(gray)
    if mean_gray < 80:
        gamma = 1.8
    elif mean_gray > 180:
        gamma = 0.6
    else:
        gamma = 1.0

    return _gamma_correction(sharpened, gamma)


def _preprocess_image_for_edges(image: np.ndarray) -> np.ndarray:
    bilateral = cv2.bilateralFilter(image, 9, 75, 75)
    lab = cv2.cvtColor(bilateral, cv2.COLOR_BGR2LAB)
    L,A,B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(L)
    img_clahe = cv2.cvtColor(cv2.merge((cl,A,B)), cv2.COLOR_LAB2BGR)
    return _unsharp_mask(img_clahe)

# ---------------- Document Detection Pipelines ---------------- #

def _auto_detect_document_corners_sharpen_adaptive(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray,9,75,75)
    sharpened = _unsharp_mask(blur)
    thresh = cv2.adaptiveThreshold(sharpened,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    return _find_quad_from_contours(morph, image)

def _auto_detect_document_corners_clahe_canny(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    edges = cv2.Canny(clahe.apply(gray),50,150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return _find_quad_from_contours(morph,image)

def _auto_detect_document_corners_bright_blur(image: np.ndarray) -> np.ndarray:
    img_gamma = _gamma_correction(image, gamma=1.8)
    gray = cv2.cvtColor(img_gamma, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(gray)
    sharp = _unsharp_mask(clahe_img)
    edges = cv2.Canny(sharp,30,120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    return _find_quad_from_contours(closed,image)

def _auto_detect_document_corners_inverse_threshold(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    _, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    return _find_quad_from_contours(morph,image)

def _auto_detect_document_corners_gradient_magnitude(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F,1,0,ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F,0,1,ksize=3)
    mag,_ = cv2.cartToPolar(grad_x,grad_y,angleInDegrees=True)
    _, thresh = cv2.threshold(mag,50,255,cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    return _find_quad_from_contours(morph,image)

def _auto_detect_document_corners_dark_object(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L = cv2.split(lab)[0]
    _, dark_mask = cv2.threshold(L,80,255,cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(dark_mask,50,150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    return _find_quad_from_contours(closed,image)

# ---------------- Candidate Detection & Selection ---------------- #

def detect_document_candidates(image: np.ndarray) -> List[np.ndarray]:
    candidates = []
    img_corrected = _adaptive_contrast_enhancement(image)
    img_preprocessed = _preprocess_image_for_edges(img_corrected)
    pipelines = [
        _auto_detect_document_corners_sharpen_adaptive,
        _auto_detect_document_corners_clahe_canny,
        _auto_detect_document_corners_bright_blur,
        _auto_detect_document_corners_inverse_threshold,
        _auto_detect_document_corners_gradient_magnitude,
        _auto_detect_document_corners_dark_object
    ]
    for func in pipelines:
        try:
            quad = func(img_preprocessed)
            if not np.allclose(quad,_full_image_quad(image),atol=1):
                candidates.append(quad)
        except Exception as e:
            print(f"Pipeline error: {e}")
    if not candidates:
        candidates.append(_full_image_quad(image))
    return candidates

def select_best_quad(image: np.ndarray, candidates: List[np.ndarray]) -> np.ndarray:
    # Placeholder: skor mantığı geliştirilebilir
    return candidates[0]

# ---------------- Component ---------------- #

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
        if img is None or img.size==0:
            raise ValueError("Input image empty or None")
        if img.dtype != np.uint8:
            img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
        if img.ndim==2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[-1]==4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def run(self):
        img_obj = Image.get_frame(img=self.image, redis_db=self.redis_db)
        if img_obj is None or img_obj.value is None:
            raise ValueError("No input image provided or failed to load")
        src_img = self._prepare_image(img_obj.value)
        candidates = detect_document_candidates(src_img)
        best_quad = select_best_quad(src_img, candidates)
        warped = _four_point_transform(src_img, best_quad)
        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)
        self.context["src_quad"] = best_quad.tolist()
        self.context["output_size"] = [warped.shape[1], warped.shape[0]]
        return build_response(context=self)

Executor(sys.argv[1]).run()
