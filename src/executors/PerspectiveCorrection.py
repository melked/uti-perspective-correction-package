import os
import sys
import cv2
import numpy as np
from typing import List, Tuple

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
    dst = np.array([[0, 0], [int(width)-1, 0], [int(width)-1, int(height)-1], [0, int(height)-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (int(width), int(height)), flags=cv2.INTER_LANCZOS4)

def _full_image_quad(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    return np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=np.float32)

# ---------------------- 2. ÖN İŞLEME ----------------------
def _preprocess(image: np.ndarray, method: str, **kwargs) -> np.ndarray:
    img = image.copy()
    if method == "bilateral":
        return cv2.bilateralFilter(img, kwargs.get("d", 9), kwargs.get("sigma", 75), kwargs.get("sigma", 75))
    elif method == "gaussian":
        return cv2.GaussianBlur(img, kwargs.get("ksize", (5,5)), 0)
    elif method == "clahe":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim==3 else img
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(gray)
    elif method == "gamma":
        table = np.array([((i/255.0)**kwargs.get("gamma",1.5))*255 for i in range(256)], dtype=np.uint8)
        img_gamma = cv2.LUT(img, table)
        return img_gamma
    elif method == "unsharp":
        blur = cv2.GaussianBlur(img, kwargs.get("ksize",(5,5)), 0)
        return cv2.addWeighted(img, 1.5, blur, -0.5, 0)
    elif method == "clahe_gamma_bilateral":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim==3 else img
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        # Gamma düzeltme
        gamma_val = kwargs.get("gamma", 1.8)
        table = np.array([((i/255.0)**gamma_val)*255 for i in range(256)], dtype=np.uint8)
        gray = cv2.LUT(gray, table)
        # Bilateral filter
        d = kwargs.get("d", 9)
        sigma = kwargs.get("sigma", 75)
        gray = cv2.bilateralFilter(gray, d, sigma, sigma)
        return gray
    else:
        return img

# ---------------------- 3. EŞİKLEME VE KENAR TESPİTİ ----------------------
def _threshold(image: np.ndarray, method: str, **kwargs) -> np.ndarray:
    img = image.copy()
    if method in ["adaptive","otsu_inv"]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim==3 else img
        if method=="adaptive":
            return cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV if kwargs.get("invert", True) else cv2.THRESH_BINARY,
                kwargs.get("block",11), kwargs.get("C",2)
            )
        else: # otsu_inv
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            return thresh
    elif method=="canny":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim==3 else img
        return cv2.Canny(gray, kwargs.get("th1",50), kwargs.get("th2",150))
    elif method=="inRange":
        if img.ndim==2:  # griyse BGR’ye çevir
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        lower = np.array(kwargs.get("lower",[0,0,180]), dtype=np.uint8)
        upper = np.array(kwargs.get("upper",[180,30,255]), dtype=np.uint8)
        return cv2.inRange(img, lower, upper)
    else:
        return img

# ---------------------- 4. MORFOLOJİK İŞLEMLER ----------------------
def _morphology(mask: np.ndarray, op: str="close", ksize=(5,5), iterations: int=1) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    if op=="close":
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    elif op=="open":
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif op=="close_open":
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    else:
        return mask

# ---------------------- 5. KONTUR VE DÖRTGEN BULMA ----------------------
def _find_quads(mask: np.ndarray, min_area_ratio: float=0.05) -> List[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    quads = []
    img_area = mask.shape[0]*mask.shape[1]
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(c) < min_area_ratio*img_area:
            break
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx)==4 and cv2.isContourConvex(approx):
            quads.append(approx.reshape(4,2).astype(np.float32))
    return quads

# ---------------------- 6. DÖRTGEN SKORLAMA ----------------------
def _score_quad(quad: np.ndarray, img_shape: Tuple[int,int]) -> float:
    area = cv2.contourArea(quad)
    img_area = img_shape[0]*img_shape[1]
    area_score = 1.0 if 0.05<area/img_area<0.95 else 0.1

    ordered = _order_points(quad)
    angles = []
    for i in range(4):
        p1, p2, p3 = ordered[(i-1)%4], ordered[i], ordered[(i+1)%4]
        v1, v2 = p1-p2, p3-p2
        angle = np.degrees(np.arccos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-6)))
        angles.append(angle)
    angle_score = np.mean([1-abs(a-90)/90 for a in angles])

    w = (np.linalg.norm(ordered[0]-ordered[1]) + np.linalg.norm(ordered[2]-ordered[3]))/2
    h = (np.linalg.norm(ordered[1]-ordered[2]) + np.linalg.norm(ordered[0]-ordered[3]))/2
    ratio_score = max(0,1-(max(w,h)/min(w,h)-1.4)/5.0)

    return area_score*0.4 + angle_score*0.4 + ratio_score*0.2

# ---------------------- 7. PIPELINE SETİ ----------------------
PIPELINES = [
    {"name":"aggressive_clahe_gamma_bilateral_canny","pre":"clahe_gamma_bilateral","pre_args":{"gamma":1.8,"d":9,"sigma":75},"thresh":"canny","thresh_args":{"th1":30,"th2":120},"morph":"close_open","morph_args":{"ksize":(5,5),"iterations":2}},
    {"name":"sharpen_adaptive","pre":"bilateral","pre_args":{"d":9,"sigma":75},"thresh":"adaptive","thresh_args":{"block":11,"C":2,"invert":True},"morph":"close_open","morph_args":{"ksize":(5,5)}},
    {"name":"clahe_canny","pre":"clahe","pre_args":{},"thresh":"canny","thresh_args":{"th1":50,"th2":150},"morph":"close","morph_args":{"ksize":(5,5)}},
    {"name":"gamma_bilateral_unsharp","pre":"gamma","pre_args":{"gamma":1.8},"thresh":"canny","thresh_args":{"th1":30,"th2":120},"morph":"close","morph_args":{"ksize":(7,7),"iterations":2}},
    {"name":"inverse_otsu","pre":"gaussian","pre_args":{"ksize":(5,5)},"thresh":"otsu_inv","thresh_args":{},"morph":"close_open","morph_args":{"ksize":(5,5)}},
    {"name":"color_segment_hsv","pre":None,"pre_args":{},"thresh":"inRange","thresh_args":{"lower":[0,0,180],"upper":[180,30,255]},"morph":"close_open","morph_args":{"ksize":(7,7)}},
]

# ---------------------- 8. ANA BİLEŞEN ----------------------
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
            raise ValueError("Input image is empty or None.")
        if img.dtype != np.uint8:
            img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
        if img.ndim==2: img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        elif img.shape[-1]==4: img=cv2.cvtColor(img,cv2.COLOR_BGRA2BGR)
        return img

    def _find_best_quad(self, image: np.ndarray) -> Tuple[np.ndarray,str,float]:
        candidates=[]
        for pipe in PIPELINES:
            try:
                proc = _preprocess(image, pipe["pre"], **pipe["pre_args"]) if pipe["pre"] else image
                thresh = _threshold(proc, pipe["thresh"], **pipe["thresh_args"])
                mask = _morphology(thresh, pipe["morph"], **pipe["morph_args"])
                quads = _find_quads(mask)
                for q in quads:
                    score = _score_quad(q, image.shape)
                    candidates.append({"quad":q,"source":pipe["name"],"score":score})
            except Exception as e:
                print(f"Pipeline '{pipe['name']}' hata verdi: {e}")

        if not candidates:
            return _full_image_quad(image), "fallback_full_image", 0.0
        best = max(candidates, key=lambda c:c["score"])
        return best["quad"], best["source"], best["score"]

    def run(self):
        img_obj = Image.get_frame(img=self.image, redis_db=self.redis_db)
        src_img = self._prepare_image(img_obj.value)

        best_quad, source, score = self._find_best_quad(src_img)
        print(f"En iyi aday '{source}' pipeline'ından {score:.2f} skorla bulundu.")

        warped = _four_point_transform(src_img, best_quad)
        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)

        self.context["src_quad"] = best_quad.tolist()
        self.context["output_size"] = [warped.shape[1], warped.shape[0]]
        self.context["best_pipeline"] = source
        self.context["confidence_score"] = score

        return build_response(context=self)

# ---------------------- 9. ÇALIŞTIRMA ----------------------
Executor(sys.argv[1]).run()
