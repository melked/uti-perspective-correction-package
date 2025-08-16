import os
import sys
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# SDK importları
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))
from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveCorrection.src.utils.response import build_response
from components.PerspectiveCorrection.src.models.PackageModel import PackageModel


# ---------------------- 1. GEOMETRİ YARDIMCILARI ----------------------

def _order_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
    return rect


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect
    width = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
    height = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))
    maxWidth, maxHeight = int(round(width)), int(round(height))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LANCZOS4)


def _full_image_quad(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)


# ---------------------- 2. PIPELINE KONFIGURASYONLARI ----------------------

PIPELINE_CONFIGS: List[Dict[str, Any]] = [
    {"name": "sharpen_adaptive",
     "preprocess": {"type": "bilateral", "d": 9, "sigma": 75},
     "threshold": {"type": "adaptive", "block": 11, "C": 2, "invert": True},
     "morphology": {"op": "close_open", "ksize": (5, 5)}},

    {"name": "clahe_canny",
     "preprocess": {"type": "clahe"},
     "threshold": {"type": "canny", "th1": 50, "th2": 150},
     "morphology": {"op": "close", "ksize": (5, 5)}},

    {"name": "bright_blur_canny",
     "preprocess": {"type": "gamma", "gamma": 1.8},
     "threshold": {"type": "canny", "th1": 30, "th2": 120},
     "morphology": {"op": "close", "ksize": (7, 7), "iter": 2}},

    {"name": "inverse_otsu",
     "preprocess": {"type": "gaussian", "ksize": (5, 5)},
     "threshold": {"type": "otsu_inv"},
     "morphology": {"op": "close_open", "ksize": (5, 5)}},

    {"name": "color_segment_hsv",
     "color_space": "hsv",
     "threshold": {"type": "inRange", "lower": [0, 0, 180], "upper": [180, 30, 255]},
     "morphology": {"op": "close_open", "ksize": (7, 7)}},

    {"name": "super_aggressive",
     "preprocess": {"type": "clahe_gamma_unsharp"},
     "threshold": {"type": "canny_adaptive", "th1": 20, "th2": 100, "adaptive_factor": 0.8},
     "morphology": {"op": "dilate_close_open", "ksize": (7, 7), "iter": 2}}
]


# ---------------------- 3. GÖRÜNTÜ İŞLEME ----------------------

def _process_pipeline(image: np.ndarray, config: Dict[str, Any]) -> Optional[np.ndarray]:
    # Renk uzayı
    if config.get("color_space") == "hsv":
        proc_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    else:
        proc_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Ön işlem
    pre = config.get("preprocess", {})
    ptype = pre.get("type")
    if ptype == "bilateral":
        proc_img = cv2.bilateralFilter(proc_img, pre["d"], pre["sigma"], pre["sigma"])
    elif ptype == "gaussian":
        proc_img = cv2.GaussianBlur(proc_img, pre["ksize"], 0)
    elif ptype == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        proc_img = clahe.apply(proc_img)
    elif ptype == "gamma":
        table = np.array([((i / 255.0) ** pre["gamma"]) * 255 for i in np.arange(256)]).astype("uint8")
        proc_img = cv2.LUT(image, table)
        proc_img = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
    elif ptype == "clahe_gamma_unsharp":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray)
        gamma = np.power(clahe_img / 255.0, 0.7) * 255
        gamma = gamma.astype(np.uint8)
        blur = cv2.GaussianBlur(gamma, (3, 3), 0)
        proc_img = cv2.addWeighted(gamma, 1.8, blur, -0.8, 0)

    # Threshold
    thresh = config.get("threshold", {})
    ttype = thresh.get("type")
    binary_mask = None
    if ttype == "adaptive":
        method = cv2.THRESH_BINARY_INV if thresh.get("invert", True) else cv2.THRESH_BINARY
        binary_mask = cv2.adaptiveThreshold(proc_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, method,
                                            thresh["block"], thresh["C"])
    elif ttype == "canny":
        binary_mask = cv2.Canny(proc_img, thresh["th1"], thresh["th2"])
    elif ttype == "otsu_inv":
        _, binary_mask = cv2.threshold(proc_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif ttype == "inRange":
        lower = np.array(thresh["lower"], dtype="uint8")
        upper = np.array(thresh["upper"], dtype="uint8")
        binary_mask = cv2.inRange(proc_img, lower, upper)
    elif ttype == "canny_adaptive":
        edges = cv2.Canny(proc_img, thresh["th1"], thresh["th2"])
        median_val = np.median(proc_img)
        factor = thresh.get("adaptive_factor", 1.0)
        _, adapt_mask = cv2.threshold(proc_img, int(median_val * factor), 255, cv2.THRESH_BINARY)
        binary_mask = cv2.bitwise_or(edges, adapt_mask)

    if binary_mask is None: return None

    # Morfoloji
    morph = config.get("morphology", {})
    if "ksize" in morph:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph["ksize"])
        iters = morph.get("iter", 1)
        op = morph["op"]
        if op == "close":
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=iters)
        elif op == "close_open":
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=iters)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=iters)
        elif op == "dilate_close_open":
            binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=iters)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=iters)

    # Konturdan dörtgen
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    img_area = image.shape[0] * image.shape[1]
    min_area = img_area * 0.05
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(c) < min_area: break
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype(np.float32)
    return None


# ---------------------- 4. SKORLAMA ----------------------

def _score_quad(quad: np.ndarray, shape: Tuple[int, int]) -> float:
    area = abs(cv2.contourArea(quad))
    img_area = shape[0] * shape[1]
    area_score = 1.0 if 0.05 < area / img_area < 0.95 else 0.1
    ordered = _order_points(quad)
    angles = []
    for i in range(4):
        p1, p2, p3 = ordered[(i - 1) % 4], ordered[i], ordered[(i + 1) % 4]
        v1, v2 = p1 - p2, p3 - p2
        angle = np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)))
        angles.append(angle)
    angle_score = np.mean([1 - abs(a - 90) / 90 for a in angles])
    w = (np.linalg.norm(ordered[0] - ordered[1]) + np.linalg.norm(ordered[2] - ordered[3])) / 2
    h = (np.linalg.norm(ordered[1] - ordered[2]) + np.linalg.norm(ordered[0] - ordered[3])) / 2
    ratio_score = max(0, 1 - (max(w, h) / (min(w, h) + 1e-6) - 1.4) / 5.0)
    return area_score * 0.4 + angle_score * 0.4 + ratio_score * 0.2


# ---------------------- 5. COMPONENT ----------------------

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
        if img is None or img.size == 0: raise ValueError("Input image empty")
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def _find_best_quad(self, img: np.ndarray) -> Tuple[np.ndarray, str, float]:
        candidates = []
        for cfg in PIPELINE_CONFIGS:
            try:
                quad = _process_pipeline(img, cfg)
                if quad is not None:
                    score = _score_quad(quad, img.shape)
                    candidates.append({"quad": quad, "source": cfg["name"], "score": score})
            except Exception as e:
                print(f"Pipeline {cfg['name']} hata: {e}")
        if not candidates:
            return _full_image_quad(img), "fallback_full_image", 0.0
        best = max(candidates, key=lambda c: c["score"])
        return best["quad"], best["source"], best["score"]

    def run(self):
        img_obj = Image.get_frame(img=self.image, redis_db=self.redis_db)
        src_img = self._prepare_image(img_obj.value)
        best_quad, source, score = self._find_best_quad(src_img)
        print(f"En iyi pipeline: {source} ({score:.2f})")
        warped = _four_point_transform(src_img, best_quad)
        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)
        self.context.update({
            "src_quad": best_quad.tolist(),
            "output_size": [warped.shape[1], warped.shape[0]],
            "best_pipeline": source,
            "confidence_score": score
        })
        return build_response(context=self)


# ---------------------- 6. ÇALIŞTIRMA ----------------------
Executor(sys.argv[1]).run()
