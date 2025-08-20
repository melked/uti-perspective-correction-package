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

# ------------------------------------------------------------------------
# 1. PARAMETRELER
# ------------------------------------------------------------------------
class Params:
    def __init__(self, config=None):
        config = config or {}
        self.resize_longest_edge = config.get("resize_longest_edge", 1000)
        self.blur_ksize = tuple(config.get("blur_ksize", (5, 5)))
        self.canny_min = config.get("canny_min", 50)
        self.canny_max = config.get("canny_max", 150)
        self.feature_max_corners = config.get("feature_max_corners", 100)
        self.feature_quality_level = config.get("feature_quality_level", 0.01)
        self.feature_min_distance = config.get("feature_min_distance", 10)
        self.contour_min_area_ratio = config.get("contour_min_area_ratio", 0.03)
        self.approx_poly_epsilon_ratio = config.get("approx_poly_epsilon_ratio", 0.02)
        self.clahe_clip = config.get("clahe_clip", 2.0)
        self.gamma = config.get("gamma", 1.2)
        self.unsharp_amount = config.get("unsharp_amount", 1.5)
        self.morph_kernel = config.get("morph_kernel", 7)
        self.hough_threshold_ratio = config.get("hough_threshold_ratio", 0.25)

# ------------------------------------------------------------------------
# 2. GEOMETRİ
# ------------------------------------------------------------------------
def _order_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> Optional[np.ndarray]:
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxWidth = int(max(widthA, widthB))
    maxHeight = int(max(heightA, heightB))
    if maxWidth <= 10 or maxHeight <= 10:
        return None
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0,maxHeight-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect,dst)
    return cv2.warpPerspective(image,M,(maxWidth,maxHeight),flags=cv2.INTER_LANCZOS4)

def _score_candidate(quad: np.ndarray, shape: tuple) -> float:
    h, w = shape[:2]
    total_area = w*h
    area = cv2.contourArea(quad.astype(np.float32))
    if area <= 0: return 0
    peri = cv2.arcLength(quad.astype(np.float32), True)
    hull = cv2.convexHull(quad.astype(np.float32))
    hull_area = cv2.contourArea(hull)
    solidity = area/hull_area if hull_area>0 else 0
    M = cv2.moments(quad.astype(np.float32))
    if M["m00"]==0: return 0
    cx = M["m10"]/M["m00"]
    cy = M["m01"]/M["m00"]
    centrality = 1.0 - (np.linalg.norm(np.array([cx,cy])-np.array([w/2,h/2]))/(max(w,h)/2))
    return 0.4*(area/total_area)+0.4*centrality+0.2*solidity

def _line_intersection(line1, line2):
    rho1,theta1 = line1
    rho2,theta2 = line2
    A = np.array([[np.cos(theta1),np.sin(theta1)],[np.cos(theta2),np.sin(theta2)]])
    b = np.array([[rho1],[rho2]])
    try:
        x0,y0 = np.linalg.solve(A,b)
        return [int(round(x0[0])),int(round(y0[0]))]
    except np.linalg.LinAlgError:
        return None

# ------------------------------------------------------------------------
# 3. ÖN İŞLEME
# ------------------------------------------------------------------------
def apply_clahe(img_gray, clip=2.0): return cv2.createCLAHE(clipLimit=clip, tileGridSize=(8,8)).apply(img_gray)
def adjust_gamma(img, gamma=1.2):
    invGamma = 1.0/gamma
    table = np.array([((i/255.0)**invGamma)*255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, table)
def unsharp_mask(img, amount=1.5):
    blur = cv2.GaussianBlur(img,(5,5),0)
    return cv2.addWeighted(img,1+amount,blur,-amount,0)

# ------------------------------------------------------------------------
# 4. BELGE KÖŞESİ BULMA
# ------------------------------------------------------------------------
def detect_document_quad(img, params: Params):
    h,w = img.shape[:2]
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = apply_clahe(gray,params.clahe_clip)
    gray = adjust_gamma(gray,params.gamma)
    gray = unsharp_mask(gray,params.unsharp_amount)
    candidates = []

    # --- goodFeaturesToTrack ---
    corners = cv2.goodFeaturesToTrack(gray,maxCorners=params.feature_max_corners,
                                      qualityLevel=params.feature_quality_level,
                                      minDistance=params.feature_min_distance)
    if corners is not None and len(corners)>=4:
        corners = np.squeeze(corners)
        hull = cv2.convexHull(corners)
        peri = cv2.arcLength(hull,True)
        approx = cv2.approxPolyDP(hull, params.approx_poly_epsilon_ratio*peri,True)
        if len(approx)==4: candidates.append(approx.reshape(4,2))

    # --- Kontur tabanlı ---
    blurred = cv2.GaussianBlur(gray, params.blur_ksize,0)
    edged = cv2.Canny(blurred, params.canny_min, params.canny_max)
    contours,_ = cv2.findContours(edged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        total_area = h*w
        for c in sorted(contours,key=cv2.contourArea,reverse=True)[:10]:
            if cv2.contourArea(c)<total_area*params.contour_min_area_ratio: continue
            peri = cv2.arcLength(c,True)
            approx = cv2.approxPolyDP(c, params.approx_poly_epsilon_ratio*peri,True)
            if len(approx)==4 and cv2.isContourConvex(approx):
                candidates.append(approx.reshape(4,2))

    # --- Hough Lines tabanlı ---
    edges = cv2.Canny(gray,(params.canny_min,params.canny_max))
    lines = cv2.HoughLines(edges,1,np.pi/180,int(min(h,w)*params.hough_threshold_ratio))
    if lines is not None:
        h_lines,v_lines=[],[]
        for line in lines:
            rho,theta = line[0]
            if theta<np.pi/4 or theta>3*np.pi/4: v_lines.append((rho,theta))
            else: h_lines.append((rho,theta))
        if len(h_lines)>=2 and len(v_lines)>=2:
            h_lines.sort(key=lambda x:x[0])
            v_lines.sort(key=lambda x:x[0])
            pts = [_line_intersection(h_lines[0],v_lines[0]),
                   _line_intersection(h_lines[0],v_lines[-1]),
                   _line_intersection(h_lines[-1],v_lines[-1]),
                   _line_intersection(h_lines[-1],v_lines[0])]
            if all(p is not None for p in pts):
                candidates.append(np.array(pts,dtype=np.float32))

    # --- En iyi aday ---
    if not candidates: return None
    best = max(candidates,key=lambda x:_score_candidate(x,img.shape))
    return best

# ------------------------------------------------------------------------
# 5. COMPONENT
# ------------------------------------------------------------------------
class PerspectiveCorrection(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request,bootstrap)
        self.context={}
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")
        self.params = Params(self.request.get_param("params",{}))

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def _prepare_image(self,img: np.ndarray) -> np.ndarray:
        if img is None or img.size==0: raise ValueError("Input image is empty or None.")
        if img.dtype!=np.uint8: img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
        if img.ndim==2: img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        elif img.shape[-1]==4: img=cv2.cvtColor(img,cv2.COLOR_BGRA2BGR)
        return img

    def run(self):
        img_obj = Image.get_frame(img=self.image,redis_db=self.redis_db)
        if img_obj is None or img_obj.value is None: raise ValueError("No input image provided or failed to load.")
        src_img = self._prepare_image(img_obj.value)
        h,w = src_img.shape[:2]
        scale = self.params.resize_longest_edge/max(h,w)
        work_img = cv2.resize(src_img,(int(w*scale),int(h*scale)))
        document_quad_scaled = detect_document_quad(work_img,self.params)
        document_quad=None
        warped=None
        if document_quad_scaled is not None:
            document_quad=document_quad_scaled/scale
            warped=_four_point_transform(src_img,document_quad)
            if warped is not None: print("Başarılı: Belge bulundu ve düzeltildi.")
        if warped is None:
            print("Tespit başarısız. Tüm görüntü kullanılıyor.")
            warped=src_img
            document_quad=np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]],dtype=np.float32)
        img_obj.value=warped
        self.image=Image.set_frame(img=img_obj,package_uID=self.uID,redis_db=self.redis_db)
        self.context["src_quad"]=document_quad.tolist()
        self.context["output_size"]=[warped.shape[1],warped.shape[0]]
        return build_response(context=self)

# ------------------------------------------------------------------------
# 6. ÇALIŞTIRICI
# ------------------------------------------------------------------------
if __name__=="__main__":
    Executor(sys.argv[1]).run()
