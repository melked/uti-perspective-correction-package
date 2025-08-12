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


# -------------------------
# Yardımcı fonksiyonlar
# -------------------------
def order_points(pts: np.ndarray) -> np.ndarray:
    """(4,2) -> ordered TL, TR, BR, BL"""
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def warp_from_quad(img: np.ndarray, quad: np.ndarray, interp=cv2.INTER_LANCZOS4) -> np.ndarray:
    rect = order_points(quad)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(round(max(widthA, widthB)))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(round(max(heightA, heightB)))
    if maxW < 10 or maxH < 10:
        return img
    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (maxW, maxH), flags=interp)


# -------------------------
# Modüler pipeline fonksiyonları
# -------------------------
def resize_preview(img: np.ndarray, max_dim: int):
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img.copy(), 1.0
    scale = max_dim / float(max(h, w))
    small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return small, scale


def preprocess_variants_preview(gray: np.ndarray):
    """Preview boyutunda hızlı varyantlar üretir: (name, binary_or_edge) listesi"""
    variants = []
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_clahe = clahe.apply(gray)
    variants.append(("clahe_canny", cv2.Canny(v_clahe, 50, 150)))
    variants.append(("gauss_canny", cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 120)))
    variants.append(("bilateral_canny", cv2.Canny(cv2.bilateralFilter(gray, 7, 75, 75), 40, 140)))
    v_adapt = cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (5, 5), 0), 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    variants.append(("adaptive", v_adapt))
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("otsu", otsu))
    # Sobel-based
    sx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    sob = cv2.convertScaleAbs(cv2.addWeighted(cv2.convertScaleAbs(sx), 0.5, cv2.convertScaleAbs(sy), 0.5, 0))
    _, sob_thr = cv2.threshold(sob, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("sobel", sob_thr))
    # Auto-canny
    med = np.median(gray)
    low = int(max(0, 0.66 * med))
    high = int(min(255, 1.33 * med))
    variants.append(("auto_canny", cv2.Canny(gray, low, high)))
    return variants


def approx_quads(edges, min_area_px=100):
    """Edges'den approx 4-kenarli konturları döndürür (list of (4,1,2) arrays)"""
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quads = []
    for c in contours:
        a = cv2.contourArea(c)
        if a < min_area_px:
            continue
        peri = cv2.arcLength(c, True)
        eps = max(3.0, 0.02 * peri)  # dynamic epsilon (px)
        approx = cv2.approxPolyDP(c, eps, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quads.append(approx.astype(np.float32))
    return quads


def score_quad_preview(quad, preview_shape, min_area_ratio=0.005, max_area_ratio=0.98):
    h, w = preview_shape[:2]
    area = cv2.contourArea(quad)
    if area < min_area_ratio * w * h or area > max_area_ratio * w * h:
        return -9999.0
    # area normalized
    area_score = area / (w * h)
    # center proximity
    cx, cy = np.mean(quad.reshape(4, 2), axis=0)
    center = np.array([w / 2, h / 2])
    center_score = 1.0 - (np.linalg.norm(np.array([cx, cy]) - center) / (np.linalg.norm(center) + 1e-9))
    # aspect via minAreaRect
    rect = cv2.minAreaRect(quad.reshape(4, 2))
    rw, rh = rect[1]
    if rw <= 0 or rh <= 0:
        ar_score = 0.0
    else:
        ar = max(rw, rh) / (min(rw, rh) + 1e-9)
        ar_score = max(0.0, 1.0 - abs(ar - 1.6) / 2.0)
    return 0.5 * area_score + 0.35 * center_score + 0.15 * ar_score


def refine_quad_fullres(quad_preview, preview_scale, full_gray, full_edges,
                        subpix_win=(5, 5), subpix_iter=40,
                        edge_overlap_thresh=0.45, min_area_ratio=0.01, max_area_ratio=0.95):
    """Preview köşelerini full-res'e ölçekle, cornerSubPix uygula ve geometrik/edge kontrolleri yap"""
    # map to full-res
    quad = (quad_preview.reshape(4, 2) / preview_scale).astype(np.float32)
    corners = quad.reshape(-1, 1, 2).astype(np.float32)

    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, subpix_iter, 0.01)
    try:
        cv2.cornerSubPix(full_gray, corners, subpix_win, (-1, -1), term)
    except Exception:
        pass

    corners2 = corners.reshape(4, 2)
    ordered = order_points(corners2)

    # angle consistency
    def angle_deg(a, b, c):
        ab = a - b
        cb = c - b
        cosang = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb) + 1e-9)
        return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

    angs = []
    for i in range(4):
        a = ordered[(i - 1) % 4]
        b = ordered[i]
        c = ordered[(i + 1) % 4]
        angs.append(angle_deg(a, b, c))
    ang_mean, ang_std = np.mean(angs), np.std(angs)
    if not (65 <= ang_mean <= 115 and ang_std < 25):
        return None

    # area bounds
    fh, fw = full_gray.shape[:2]
    full_area = cv2.contourArea(ordered.reshape(4, 1, 2))
    if full_area < min_area_ratio * fw * fh or full_area > max_area_ratio * fw * fh:
        return None

    # edge overlap check
    mask = np.zeros_like(full_gray)
    cv2.drawContours(mask, [ordered.reshape(4, 1, 2).astype(np.int32)], -1, 255, 3)
    overlap = cv2.countNonZero(cv2.bitwise_and(mask, full_edges))
    total = cv2.countNonZero(mask)
    if total == 0 or (overlap / (total + 1e-9)) < edge_overlap_thresh:
        return None

    return ordered  # (4,2) float32


# -------------------------
# Ana fonksiyon (modüler, kısa)
# -------------------------
def correct_perspective_advanced(image: np.ndarray, params: dict = None) -> np.ndarray:
    """
    Modüler preview->refine pipeline.
    params örnek:
    {
      "preview_max_dim": 900,
      "top_k": 3,
      "subpix_win": (5,5),
      "subpix_iter": 40,
      "edge_overlap_thresh": 0.45
    }
    """
    if params is None:
        params = {}
    PREVIEW_MAX = params.get("preview_max_dim", 900)
    TOP_K = params.get("top_k", 3)
    SUBPIX_WIN = tuple(params.get("subpix_win", (5, 5)))
    SUBPIX_ITERS = params.get("subpix_iter", 40)
    EDGE_OVERLAP = params.get("edge_overlap_thresh", 0.45)

    # Preview resize
    preview_img, scale = resize_preview(image, PREVIEW_MAX)
    preview_gray = cv2.cvtColor(preview_img, cv2.COLOR_BGR2GRAY)

    # generate variants and collect quads with preview-scores
    variants = preprocess_variants_preview(preview_gray)
    preview_candidates = []
    for name, edges in variants:
        quads = approx_quads(edges, min_area_px=20)
        for q in quads:
            s = score_quad_preview(q, preview_img.shape)
            if s > 0:
                preview_candidates.append((s, q, name))
    preview_candidates.sort(key=lambda x: x[0], reverse=True)

    # fallback: if nothing found, try full-res otsu approx
    if not preview_candidates:
        gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thr = cv2.threshold(gray_full, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image
        big = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(big, True)
        approx = cv2.approxPolyDP(big, 0.02 * peri, True)
        if len(approx) == 4:
            return warp_from_quad(image, approx.reshape(4,2))
        return image

    # full-res edges for overlap testing
    gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    full_edges = cv2.Canny(cv2.GaussianBlur(gray_full, (3, 3), 0), 50, 150)

    # refine top-K candidates on full-res
    refined = []
    for i in range(min(TOP_K, len(preview_candidates))):
        score_p, quad_preview, vname = preview_candidates[i]
        qref = refine_quad_fullres(quad_preview, scale, gray_full, full_edges,
                                   subpix_win=SUBPIX_WIN, subpix_iter=SUBPIX_ITERS,
                                   edge_overlap_thresh=EDGE_OVERLAP)
        if qref is not None:
            # final score: area + center proximity
            area = cv2.contourArea(qref.reshape(4,1,2))
            center = np.mean(qref, axis=0)
            w_full, h_full = image.shape[1], image.shape[0]
            center_dist = np.linalg.norm(center - np.array([w_full/2, h_full/2]))
            center_score = 1 - (center_dist / (np.linalg.norm(np.array([w_full/2, h_full/2])) + 1e-9))
            final_score = area * 0.6 + center_score * 0.4
            refined.append((final_score, qref))
    if refined:
        best = max(refined, key=lambda x: x[0])[1]
        return warp_from_quad(image, best)
    # else fallback to best preview mapped to full-res
    best_preview = preview_candidates[0][1].reshape(4,2).astype(np.float32)
    mapped = (best_preview / scale).astype(np.float32)
    return warp_from_quad(image, mapped)


# -------------------------
# Component sınıfı (senin yapıya uyumlu)
# -------------------------
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
            self.context["output_size"] = [warped.shape[1], warped.shape[0]]
        except Exception:
            pass
        return build_response(context=self)


if __name__ == "__main__":
    Executor(sys.argv[1]).run()
