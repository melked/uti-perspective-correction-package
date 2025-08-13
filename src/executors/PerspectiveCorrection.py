import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveCorrection.src.utils.response import build_response
from components.PerspectiveCorrection.src.models.PackageModel import PackageModel

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def preprocess_variants(img):
    variants = []
    canny_params = [(30,100),(50,150),(100,200)]
    blur_kernels = [(3,3),(5,5),(7,7)]
    gamma_vals = [0.6,0.8,1.2]

    def clahe_fn(i):
        clahe_obj = cv2.createCLAHE(clipLimit=i, tileGridSize=(8,8))
        return clahe_obj.apply(img)

    def unsharp_mask_fn(i):
        gaussian = cv2.GaussianBlur(img,(i,i),10.0)
        return cv2.addWeighted(img,1.5,gaussian,-0.5,0)

    # 1️⃣ Normal CLAHE + Canny
    for low, high in canny_params:
        v = clahe_fn(2.0)
        edges = cv2.Canny(v,low,high)
        variants.append((f"normal_{low}_{high}", edges))

    # 2️⃣ Agresif CLAHE + Unsharp + Canny
    for low, high in canny_params:
        v = unsharp_mask_fn(9)
        edges = cv2.Canny(v,low,high)
        variants.append((f"agresif_{low}_{high}", edges))

    # 3️⃣ Yumuşak Gaussian Blur + Adaptive Threshold
    for k in blur_kernels:
        v = cv2.GaussianBlur(img,k,0)
        thr = cv2.adaptiveThreshold(v,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        variants.append((f"yumusak_{k[0]}",thr))

    # 4️⃣ Gamma Correction + CLAHE + Canny
    for g in gamma_vals:
        v = clahe_fn(2.0)
        gamma_corr = np.array(np.power(v/255.0,g)*255,dtype=np.uint8)
        edges = cv2.Canny(gamma_corr,50,150)
        variants.append((f"gamma_{g}",edges))

    # 5️⃣ CLAHE + Otsu
    v = clahe_fn(2.0)
    otsu = cv2.threshold(v,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    variants.append(("otsu",otsu))

    # 6️⃣ Morph Gradient
    for size in [3,5]:
        v = clahe_fn(2.0)
        morph = cv2.morphologyEx(v,cv2.MORPH_GRADIENT,np.ones((size,size),np.uint8))
        variants.append((f"morph_{size}",morph))

    # 7️⃣ Bilateral + Canny
    for d in [7,9]:
        v = cv2.bilateralFilter(img,d,75,75)
        edges = cv2.Canny(v,50,150)
        variants.append((f"bilateral_{d}",edges))

    # 8️⃣ Top-hat Morph + Canny
    for size in [3,5]:
        kernel = np.ones((size,size),np.uint8)
        v = cv2.morphologyEx(img,cv2.MORPH_TOPHAT,kernel)
        edges = cv2.Canny(v,50,150)
        variants.append((f"tophat_{size}",edges))

    return variants

def find_corners_from_edges(edges):
    # Hough Lines ile çizgi tespiti
    lines = cv2.HoughLinesP(edges,1,np.pi/180,80,minLineLength=30,maxLineGap=10)
    if lines is None:
        return None
    # Çizgilerden kesişim noktalarını bul
    points = []
    for i,l1 in enumerate(lines):
        for j,l2 in enumerate(lines):
            if i>=j: continue
            xdiff = np.array([l1[0][0]-l1[0][2], l2[0][0]-l2[0][2]])
            ydiff = np.array([l1[0][1]-l1[0][3], l2[0][1]-l2[0][3]])
            def det(a,b): return a[0]*b[1]-a[1]*b[0]
            div = det(xdiff,ydiff)
            if div==0: continue
            d = (det([l1[0][0],l1[0][1]],[l1[0][2],l1[0][3]]),det([l2[0][0],l2[0][1]],[l2[0][2],l2[0][3]]))
            x = det(d,xdiff)/div
            y = det(d,ydiff)/div
            if 0<=x<edges.shape[1] and 0<=y<edges.shape[0]:
                points.append([x,y])
    if len(points)<4:
        return None
    points = np.array(points,dtype=np.float32)
    # Köşe iyileştirme
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    cv2.cornerSubPix(edges,points,(5,5),(-1,-1),criteria)
    # 4 en iyi köşe seçimi (basit: en uzak 4 nokta)
    from scipy.spatial import distance
    dists = distance.cdist(points,points)
    sumd = dists.sum(axis=1)
    idxs = np.argsort(sumd)[-4:]
    return points[idxs]

def correct_perspective_advanced(image, params=None):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    variants = preprocess_variants(gray)
    best_rect = None
    max_score = -1
    for name,edges in variants:
        corners = find_corners_from_edges(edges)
        if corners is None: continue
        # Basit scoring: alan + merkeze yakınlık
        tl,tr,br,bl = order_points(corners)
        w = int(max(np.linalg.norm(br-bl),np.linalg.norm(tr-tl)))
        h = int(max(np.linalg.norm(tr-br),np.linalg.norm(tl-bl)))
        score = w*h - abs(np.mean(corners[:,0])-gray.shape[1]/2) - abs(np.mean(corners[:,1])-gray.shape[0]/2)
        if score>max_score:
            max_score=score
            best_rect = np.array([tl,tr,br,bl],dtype=np.float32)
    if best_rect is None:
        return image
    dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]],dtype=np.float32)
    M = cv2.getPerspectiveTransform(best_rect,dst)
    warped = cv2.warpPerspective(image,M,(w,h))
    return warped

class PerspectiveCorrection(Component):
    def __init__(self,request,bootstrap):
        super().__init__(request,bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")
        self.params = self.request.get_param("params",None)

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def run(self):
        img = Image.get_frame(img=self.image, redis_db=self.redis_db)
        img_np = img.value
        if img_np.dtype!=np.uint8:
            if img_np.max()<=1.0:
                img_np = (img_np*255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)
        result_img = correct_perspective_advanced(img_np,self.params)
        img.value = np.array(result_img)
        self.image = Image.set_frame(img=img,package_uID=self.uID,redis_db=self.redis_db)
        return build_response(context=self)

if __name__=="__main__":
    Executor(sys.argv[1]).run()
