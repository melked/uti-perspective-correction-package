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

# --- PARAMETRELER ---

class Params:
    def __init__(self,
                 clahe_clip=3.0,
                 clahe_grid=(8, 8),
                 gamma_target=0.5,
                 sharpen_alpha=1.5,
                 sharpen_beta=-0.5,
                 sharpen_sigma=3,
                 blur_ksize=(5, 5),
                 canny_min=50,
                 canny_max=150,
                 morph_kernel_size=5,
                 hough_rho=1,
                 hough_theta=np.pi/180,
                 hough_threshold=100,
                 cluster_eps=20,
                 cluster_min_samples=1,
                 min_contour_area=1000,
                 approx_poly_epsilon_ratio=0.02,
                 max_good_features=20,
                 good_feature_quality=0.01,
                 good_feature_min_dist=20):
        self.clahe_clip = clahe_clip
        self.clahe_grid = clahe_grid
        self.gamma_target = gamma_target
        self.sharpen_alpha = sharpen_alpha
        self.sharpen_beta = sharpen_beta
        self.sharpen_sigma = sharpen_sigma
        self.blur_ksize = blur_ksize
        self.canny_min = canny_min
        self.canny_max = canny_max
        self.morph_kernel_size = morph_kernel_size
        self.hough_rho = hough_rho
        self.hough_theta = hough_theta
        self.hough_threshold = hough_threshold
        self.cluster_eps = cluster_eps
        self.cluster_min_samples = cluster_min_samples
        self.min_contour_area = min_contour_area
        self.approx_poly_epsilon_ratio = approx_poly_epsilon_ratio
        self.max_good_features = max_good_features
        self.good_feature_quality = good_feature_quality
        self.good_feature_min_dist = good_feature_min_dist


# --- FONKSİYONLAR ---

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def automatic_gamma(img, target=0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
    mean = gray.mean()
    if mean <= 0:
        return 1.0
    gamma = np.log(target) / np.log(mean)
    return max(0.3, min(gamma, 3.0))

def sharpen(img, alpha=1.5, beta=-0.5, sigma=3):
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, alpha, blur, beta, 0)

def preprocess(img, params: Params):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=params.clahe_clip, tileGridSize=params.clahe_grid)
    enhanced = clahe.apply(gray)
    gamma_val = automatic_gamma(img, target=params.gamma_target)
    table = np.array([(i / 255.0) ** (1.0 / gamma_val) * 255 for i in range(256)]).astype(np.uint8)
    gamma_corrected = cv2.LUT(enhanced, table)
    sharpened = sharpen(gamma_corrected, alpha=params.sharpen_alpha, beta=params.sharpen_beta, sigma=params.sharpen_sigma)
    blurred = cv2.GaussianBlur(sharpened, params.blur_ksize, 0)
    return blurred

def line_intersections(lines):
    def compute_intersection(l1, l2):
        rho1, theta1 = l1
        rho2, theta2 = l2
        A = np.array([[np.cos(theta1), np.sin(theta1)],
                      [np.cos(theta2), np.sin(theta2)]])
        b = np.array([[rho1], [rho2]])
        det = np.linalg.det(A)
        if abs(det) < 1e-10:
            return None
        x0, y0 = np.linalg.solve(A, b)
        return [int(np.round(x0)), int(np.round(y0))]

    points = []
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            pt = compute_intersection(lines[i], lines[j])
            if pt is not None:
                points.append(pt)
    return np.array(points)

def cluster_points(points, eps=20, min_samples=1):
    if len(points) == 0:
        return np.array([])
    from sklearn.cluster import DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    clustered_points = []
    for label in set(labels):
        pts = points[labels == label]
        mean_pt = pts.mean(axis=0)
        clustered_points.append(mean_pt)
    return np.array(clustered_points)

def detect_corners_by_lines(img, params: Params):
    edges = cv2.Canny(img, params.canny_min, params.canny_max)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((params.morph_kernel_size, params.morph_kernel_size), np.uint8))
    lines = cv2.HoughLines(edges, params.hough_rho, params.hough_theta, params.hough_threshold)
    if lines is None or len(lines) < 2:
        return None
    lines = lines[:, 0, :]
    points = line_intersections(lines)
    if len(points) == 0:
        return None
    clustered = cluster_points(points, eps=params.cluster_eps, min_samples=params.cluster_min_samples)
    if len(clustered) < 4:
        return None
    center = clustered.mean(axis=0)
    dists = np.linalg.norm(clustered - center, axis=1)
    idxs = np.argsort(dists)[-4:]
    corners = clustered[idxs]
    return order_points(corners.astype(np.float32))

def detect_corners(img, params: Params):
    corners = detect_corners_by_lines(img, params)
    if corners is not None:
        return corners
    # fallback
    corners = cv2.goodFeaturesToTrack(img, maxCorners=params.max_good_features,
                                      qualityLevel=params.good_feature_quality,
                                      minDistance=params.good_feature_min_dist)
    if corners is None or len(corners) < 4:
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, params.approx_poly_epsilon_ratio * peri, True)
            if len(approx) == 4 and cv2.contourArea(approx) > params.min_contour_area:
                return order_points(approx.reshape(4, 2))
        return None
    corners = np.squeeze(corners)
    if len(corners) > 4:
        center = corners.mean(axis=0)
        dists = np.linalg.norm(corners - center, axis=1)
        idxs = np.argsort(dists)[-4:]
        corners = corners[idxs]
    return order_points(corners)

def four_point_transform(img, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))
    dst = np.array([[0,0],
                    [maxWidth-1,0],
                    [maxWidth-1,maxHeight-1],
                    [0,maxHeight-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return warped

def correct_perspective(img, params: Params, return_intermediate=False):
    pre = preprocess(img, params)
    edges = cv2.Canny(pre, params.canny_min, params.canny_max)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((params.morph_kernel_size, params.morph_kernel_size), np.uint8))
    corners = detect_corners(edges, params)
    if corners is None:
        raise ValueError("Belge köşeleri bulunamadı.")
    warped = four_point_transform(img, corners)

    if return_intermediate:
        # Orijinal, Ön işlenmiş, Kenarlar, Köşeler işaretli
        draw_img = img.copy()
        for x, y in corners.astype(int):
            cv2.circle(draw_img, (x, y), 15, (0, 255, 0), 5)
        return (PILImage.fromarray(warped),
                PILImage.fromarray(pre),
                PILImage.fromarray(edges),
                PILImage.fromarray(draw_img))
    return PILImage.fromarray(warped)


# --- Entegrasyon sınıfı ---

class PerspectiveCorrection(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")
        # Burada parametreleri JSON'dan da alabiliriz:
        self.params = Params()

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

        # return_intermediate parametresi istekten alınabilir
        # örn: self.request.get_param("returnIntermediate", False)
        return_intermediate = self.request.get_param("returnIntermediate", False)

        if return_intermediate:
            warped, pre, edges, draw = correct_perspective(img_np, self.params, return_intermediate=True)
            # Örneğin ara görüntüleri dizide döndürelim:
            img_warped = Image.set_frame(img=Image(warped), package_uID=self.uID, redis_db=self.redis_db)
            img_pre = Image.set_frame(img=Image(pre), package_uID=self.uID, redis_db=self.redis_db)
            img_edges = Image.set_frame(img=Image(edges), package_uID=self.uID, redis_db=self.redis_db)
            img_draw = Image.set_frame(img=Image(draw), package_uID=self.uID, redis_db=self.redis_db)
            return build_response(context=self,
                                  extra={
                                    "warped": img_warped,
                                    "preprocessed": img_pre,
                                    "edges": img_edges,
                                    "corners_marked": img_draw
                                  })
        else:
            result_img = correct_perspective(img_np, self.params, return_intermediate=False)
            img.value = np.array(result_img)
            self.image = Image.set_frame(img=img, package_uID=self.uID, redis_db=self.redis_db)
            return build_response(context=self)


if __name__ == "__main__":
    Executor(sys.argv[1]).run()
