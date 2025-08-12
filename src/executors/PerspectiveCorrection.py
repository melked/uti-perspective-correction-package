import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveCorrection.src.utils.response import build_response
from components.PerspectiveCorrection.src.models.PackageModel import PackageModel


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def correct_perspective_advanced(image: np.ndarray, params: dict = None) -> np.ndarray:
    """
    Hızlı preview tabanlı candidate -> full-refine pipeline.
    params (opsiyonel): {
      "preview_max_dim": 900,
      "max_refine": 3,
      "corner_subpix_win": (5,5),
      "corner_subpix_iter": 40
    }
    """

    if params is None:
        params = {}
    PREVIEW_MAX = params.get("preview_max_dim", 900)
    MAX_REFINE = params.get("max_refine", 3)
    SUBPIX_WIN = params.get("corner_subpix_win", (5, 5))
    SUBPIX_ITERS = params.get("corner_subpix_iter", 40)
    EDGE_OVERLAP_THRESH = params.get("edge_overlap_thresh", 0.45)
    MIN_AREA_RATIO = params.get("min_area_ratio", 0.01)   # preview area ratio
    MAX_AREA_RATIO = params.get("max_area_ratio", 0.95)

    def resize_for_preview(img, max_dim=PREVIEW_MAX):
        h, w = img.shape[:2]
        scale = 1.0
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            img_small = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        else:
            img_small = img.copy()
        return img_small, scale

    def generate_preview_variants(gray):
        """Hızlı, düşük maliyetli varyasyon seti (preview üzerinde)."""
        variants = []
        # CLAHE + Canny
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        v1 = clahe.apply(gray)
        variants.append(("clahe_canny", cv2.Canny(v1, 50, 150)))

        # Gaussian blur + Canny (düşük eşik)
        v2 = cv2.GaussianBlur(gray, (5,5), 0)
        variants.append(("blur_canny", cv2.Canny(v2, 30, 120)))

        # Bilateral (kenar koruyucu) + Canny
        v3 = cv2.bilateralFilter(gray, 7, 75, 75)
        variants.append(("bilateral_canny", cv2.Canny(v3, 40, 140)))

        # Adaptive threshold (yumuşak belgeler)
        v4 = cv2.adaptiveThreshold(cv2.GaussianBlur(gray,(5,5),0), 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        variants.append(("adaptive_thresh", v4))

        # Otsu
        _, v5 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(("otsu", v5))

        # Sobel (kenar güçlendirme) + threshold
        sx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
        sob = cv2.convertScaleAbs(cv2.addWeighted(cv2.convertScaleAbs(sx), 0.5, cv2.convertScaleAbs(sy), 0.5, 0))
        _, v6 = cv2.threshold(sob, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(("sobel_thresh", v6))

        # CLAHE + adaptive inverted (parlak arka plan)
        cla = clahe.apply(gray)
        v7 = cv2.adaptiveThreshold(cla, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
        variants.append(("clahe_adaptive_inv", v7))

        # Fast Canny with auto thresholds (preview median)
        v = np.median(gray)
        lower = int(max(0, 0.66 * v))
        upper = int(min(255, 1.33 * v))
        variants.append(("auto_canny", cv2.Canny(gray, lower, upper)))

        return variants

    def approx_quads_from_edges(edges):
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        quads = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 10:  # very tiny
                continue
            peri = cv2.arcLength(cnt, True)
            # dynamic epsilon: base 0.02 scaled by perimeter
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                quads.append(approx)
        return quads

    def score_preview_quad(quad, edges, img_shape):
        """Preview üzerinde hızlı skor: alan, merkez yakınlığı, bounding aspect"""
        area = cv2.contourArea(quad)
        h, w = img_shape[:2]
        if area < (MIN_AREA_RATIO * w * h) or area > (MAX_AREA_RATIO * w * h):
            return -9999.0
        pts = quad.reshape(4, 2)
        # area score normalized to [0,1]
        area_score = area / (w*h)
        # center proximity
        cx, cy = np.mean(pts, axis=0)
        center = np.array([w/2, h/2])
        dist = np.linalg.norm(np.array([cx, cy]) - center)
        center_score = 1.0 - (dist / np.linalg.norm(center))
        # aspect ratio approximation from minAreaRect
        rect = cv2.minAreaRect(pts.astype(np.float32))
        (rw, rh) = rect[1]
        if rw == 0 or rh == 0:
            ar_score = 0.0
        else:
            ar = max(rw, rh) / (min(rw, rh) + 1e-6)
            # ideal documents often have ar between ~1.2 and 2.5; map to score
            ar_score = max(0.0, 1.0 - abs(ar - 1.6) / 2.0)
        return area_score * 0.5 + center_score * 0.35 + ar_score * 0.15

    def refine_on_full_res(quad_preview, preview_scale, full_gray, full_edges):
        """
        quad_preview: coords on preview image; scale back to full-res coordinates
        Do: map to full-res, run cornerSubPix, check edge overlap on full edges,
            validate geometry (angles, aspect), return refined quad or None.
        """
        # Map to full-res
        quad = (quad_preview.reshape(4,2).astype(np.float32) / preview_scale)
        quad = quad.reshape(4,1,2).astype(np.float32)

        # Prepare for subpixel: need float32 single-channel image
        # cornerSubPix expects corners as (N,1,2)
        # Initial corners must be float32
        corners = quad.astype(np.float32)

        # cornerSubPix params
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, SUBPIX_ITERS, 0.01)
        try:
            cv2.cornerSubPix(full_gray, corners, SUBPIX_WIN, (-1,-1), term)
        except Exception:
            # fallback: skip subpix if it fails
            pass

        corners2 = corners.reshape(4,2)

        ordered = order_points(corners2)

        def angle(a,b,c):
            ab = a - b
            cb = c - b
            cos_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb) + 1e-9)
            return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

        pts = ordered
        angs = []
        for i in range(4):
            a = pts[(i-1)%4]
            b = pts[i]
            c = pts[(i+1)%4]
            angs.append(angle(a,b,c))
        ang_std = np.std(angs)
        ang_mean = np.mean(angs)

        # angle filter: mean near 90 and std small
        if not (70 <= ang_mean <= 110 and ang_std < 20):
            # reject if angles too far from rectangle
            return None

        # check area bounds on full-res
        full_h, full_w = full_gray.shape[:2]
        full_area = cv2.contourArea(ordered.reshape(4,1,2))
        if full_area < (MIN_AREA_RATIO * full_w * full_h) or full_area > (MAX_AREA_RATIO * full_w * full_h):
            return None

        # Edge overlap on full-res: draw mask of quad and check overlap with Canny edges
        mask = np.zeros_like(full_gray)
        cv2.drawContours(mask, [ordered.reshape(4,1,2).astype(np.int32)], -1, 255, 3)
        overlap = cv2.countNonZero(cv2.bitwise_and(mask, full_edges))
        total = cv2.countNonZero(mask)
        if total == 0 or (overlap / (total + 1e-9)) < EDGE_OVERLAP_THRESH:
            return None

        return ordered.reshape(4,2)

    preview_img, scale = resize_for_preview(image, PREVIEW_MAX)
    preview_gray = cv2.cvtColor(preview_img, cv2.COLOR_BGR2GRAY)

    variants = generate_preview_variants(preview_gray)
    preview_candidates = []  # list of tuples (score, quad_on_preview, variant_name)

    for name, edges in variants:
        quads = approx_quads_from_edges(edges)
        for q in quads:
            s = score_preview_quad(q, edges, preview_img.shape)
            if s > 0:  # simple threshold to remove tiny ones
                preview_candidates.append((s, q, name))

    # sort descending by score and keep top K (fast)
    preview_candidates = sorted(preview_candidates, key=lambda x: x[0], reverse=True)
    if not preview_candidates:
        # fallback to full-image Otsu approx if nothing found
        gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray_full, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image
        big = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(big, True)
        approx = cv2.approxPolyDP(big, 0.02 * peri, True)
        if len(approx) == 4:
            best_full = approx.reshape(4,2).astype(np.float32)
            rect = order_points(best_full)
            tl, tr, br, bl = rect
            # warp and return
            widthA = np.linalg.norm(br - bl)
            widthB = np.linalg.norm(tr - tl)
            maxW = int(round(max(widthA, widthB)))
            heightA = np.linalg.norm(tr - br)
            heightB = np.linalg.norm(tl - bl)
            maxH = int(round(max(heightA, heightB)))
            dst = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(rect, dst)
            return cv2.warpPerspective(image, M, (maxW, maxH))
        return image

    top_k = min(MAX_REFINE, len(preview_candidates))
    gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    full_edges = cv2.Canny(cv2.GaussianBlur(gray_full, (3,3), 0), 50, 150)

    refined_results = []
    for i in range(top_k):
        score_p, quad_preview, variant_name = preview_candidates[i]
        refined = refine_on_full_res(quad_preview, scale, gray_full, full_edges)
        if refined is not None:
            # final area-based and angle-based score on full-res
            final_area = cv2.contourArea(refined.reshape(4,1,2))
            final_center = np.mean(refined, axis=0)
            w_full, h_full = image.shape[1], image.shape[0]
            center_dist = np.linalg.norm(final_center - np.array([w_full/2, h_full/2]))
            center_score = 1 - (center_dist / np.linalg.norm(np.array([w_full/2, h_full/2])))
            refined_results.append((final_area * 0.6 + center_score * 0.4, refined))

    if not refined_results:
        # nothing survived full-res refine: fallback to best preview candidate but scale up corners
        best_preview = preview_candidates[0][1].reshape(4,2).astype(np.float32)
        mapped = (best_preview / scale).astype(np.float32)
        rect = order_points(mapped)
    else:
        # choose best refined result
        best_idx = max(range(len(refined_results)), key=lambda k: refined_results[k][0])
        rect = order_points(refined_results[best_idx][1])

    # do final warp
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(round(max(widthA, widthB)))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(round(max(heightA, heightB)))

    if maxWidth < 10 or maxHeight < 10:
        return image

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LANCZOS4)

    return warped

class PerspectiveCorrection(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")
        self.params = self.request.get_param("params", None)
        self.context = {}

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def _prepare_image(self, img):
        if img is None or img.size == 0:
            raise ValueError("Input image is empty or None.")
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def run(self):
        img_obj = Image.get_frame(img=self.image, redis_db=self.redis_db)
        if img_obj is None or img_obj.value is None:
            raise ValueError("No input image provided or failed to load.")

        src = self._prepare_image(img_obj.value)

        warped = correct_perspective_advanced(src, self.params)

        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)

        try:
            self.context["src_quad"] = None
            self.context["output_size"] = [warped.shape[1], warped.shape[0]]
        except Exception:
            pass

        return build_response(context=self)

if __name__ == "__main__":
    Executor(sys.argv[1]).run()
