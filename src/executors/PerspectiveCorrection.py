import os
import sys
import cv2
import numpy as np
import math
from typing import Optional, List, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveCorrection.src.utils.response import build_response
from components.PerspectiveCorrection.src.models.PackageModel import PackageModel


# =============================================================================
# Geometri & Yardımcılar
# =============================================================================
def _order_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # tl
    rect[2] = pts[np.argmax(s)]      # br
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # tr
    rect[3] = pts[np.argmax(diff)]   # bl
    return rect


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> Optional[np.ndarray]:
    rect = _order_points(pts)
    tl, tr, br, bl = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(br - tr)
    heightB = np.linalg.norm(bl - tl)
    maxW = int(max(widthA, widthB))
    maxH = int(max(heightA, heightB))
    if maxW <= 10 or maxH <= 10:
        return None
    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxW, maxH), flags=cv2.INTER_LANCZOS4)


def _quad_area(quad: np.ndarray) -> float:
    q = quad.reshape(-1, 2)
    return 0.5 * abs(np.dot(q[:, 0], np.roll(q[:, 1], -1)) - np.dot(q[:, 1], np.roll(q[:, 0], -1)))


def _aspect_ratio(quad: np.ndarray) -> float:
    tl, tr, br, bl = _order_points(quad)
    w = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    h = max(np.linalg.norm(br - tr), np.linalg.norm(bl - tl))
    if min(w, h) < 1:
        return 1.0
    return max(w / h, h / w)  # >= 1


def _angles_of_quad(quad: np.ndarray) -> List[float]:
    q = _order_points(quad)
    def angle(a, b, c):
        ba = a - b
        bc = c - b
        cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        cosang = np.clip(cosang, -1.0, 1.0)
        return math.degrees(math.acos(cosang))
    tl, tr, br, bl = q
    return [
        angle(bl, tl, tr),
        angle(tl, tr, br),
        angle(tr, br, bl),
        angle(br, bl, tl),
    ]


def _edge_angles(quad: np.ndarray) -> Tuple[float, float, float, float]:
    q = _order_points(quad)
    def deg(p1, p2):
        v = p2 - p1
        return math.degrees(math.atan2(v[1], v[0]))
    return (deg(q[0], q[1]), deg(q[1], q[2]), deg(q[2], q[3]), deg(q[3], q[0]))


def _verify_quad(quad: np.ndarray, image_shape: tuple) -> Tuple[bool, dict]:
    h, w = image_shape[:2]
    area = _quad_area(quad)
    if area <= 0:
        return False, {"reason": "area<=0"}
    area_ratio = area / float(w * h)
    if not (0.03 <= area_ratio <= 0.98):
        return False, {"reason": f"area_ratio:{area_ratio:.3f}"}
    ar = _aspect_ratio(quad)
    aspect_ok = 1.0 <= ar <= 2.8

    angs = _angles_of_quad(quad)
    angle_dev = float(np.mean([abs(a - 90.0) for a in angs]))

    e0, e1, e2, e3 = _edge_angles(quad)
    def pdiff(a, b):
        d = abs(a - b) % 180.0
        return min(d, 180.0 - d)
    parallel_h = pdiff(e0, e2)
    parallel_v = pdiff(e1, e3)

    ok = aspect_ok and angle_dev <= 20.0 and parallel_h <= 18.0 and parallel_v <= 18.0
    return ok, {
        "area_ratio": area_ratio,
        "aspect_ratio": ar,
        "angle_dev": angle_dev,
        "parallel_h": parallel_h,
        "parallel_v": parallel_v
    }


def _score_candidate(quad: np.ndarray, image_shape: tuple) -> float:
    h, w = image_shape[:2]
    area = _quad_area(quad)
    area_ratio = area / float(w * h)
    area_term = np.clip((area_ratio - 0.02) / 0.5, 0, 1)

    ok, m = _verify_quad(quad, image_shape)
    angle_pen = np.clip(m["angle_dev"] / 35.0, 0, 1)
    parallel_pen = np.clip((m["parallel_h"] + m["parallel_v"]) / 70.0, 0, 1)

    # merkeze yakınlık
    M = cv2.moments(quad.reshape(-1, 1, 2).astype(np.float32))
    if M["m00"] != 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        cent = 1.0 - (np.linalg.norm([cx - w/2, cy - h/2]) / (max(w, h)/2))
        cent = float(np.clip(cent, 0, 1))
    else:
        cent = 0.0

    score = 0.55 * area_term + 0.25 * cent + 0.20 * (1.0 - 0.5 * angle_pen - 0.5 * parallel_pen)
    if ok:
        score += 0.05
    return float(np.clip(score, 0.0, 1.0))


def _auto_canny_thresholds(gray: np.ndarray, sigma: float) -> Tuple[int, int]:
    v = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    lower = max(5, lower)
    upper = max(lower + 5, upper)
    return lower, upper


def _resize_long_edge(img: np.ndarray, max_long: int = 1600) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    L = max(h, w)
    if L <= max_long:
        return img, 1.0
    s = max_long / float(L)
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA), s


# =============================================================================
# Preprocess Varyantları
# =============================================================================
def _clahe_bgr(img: np.ndarray, clip=3.0, tiles=(8, 8)) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tiles)
    L = clahe.apply(L)
    return cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)


def _gamma_bgr(img: np.ndarray, gamma: float) -> np.ndarray:
    gamma = float(np.clip(gamma, 0.3, 3.0))
    table = np.array([(i/255.0) ** gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)


def _unsharp(img: np.ndarray, s=1.2, amount=1.35) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (0, 0), s)
    return cv2.addWeighted(img, amount, blur, -(amount - 1.0), 0)


def _bilateral(img: np.ndarray, d=7, sc=50, ss=50) -> np.ndarray:
    return cv2.bilateralFilter(img, d, sc, ss)


def _background_flatten(gray: np.ndarray, k: int) -> np.ndarray:
    k = max(3, k | 1)
    bg = cv2.GaussianBlur(gray, (k, k), 0)
    flat = cv2.divide(gray, bg, scale=255)
    return (np.clip(flat, 0, 255)).astype("uint8")


def _preprocess_variants(img: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """Birçok kombinasyon üretir; maliyeti dengeli tutmak için sayıyı sınırladık."""
    base = img.copy()
    out = [("base", base)]
    v1 = _clahe_bgr(base, 2.5, (8, 8))
    v2 = _gamma_bgr(v1, 0.75)
    v3 = _unsharp(v2, 1.3, 1.4)
    out += [("clahe_g075_sharp", v3)]

    v4 = _clahe_bgr(base, 3.0, (8, 8))
    v5 = _gamma_bgr(v4, 1.25)
    v6 = _unsharp(v5, 1.1, 1.25)
    out += [("clahe_g125_sharp", v6)]

    v7 = _bilateral(base, 7, 60, 60)
    v7 = _unsharp(v7, 1.2, 1.3)
    out += [("bilateral_sharp", v7)]

    # hafif oversharp (kötü fokus senaryosu)
    v8 = _gamma_bgr(_clahe_bgr(base, 2.0, (8, 8)), 0.9)
    v8 = _unsharp(v8, 1.6, 1.55)
    out += [("over_sharp", v8)]

    return out


# =============================================================================
# Aday Üreticiler (tamamen algoritmik)
# =============================================================================
def _contour_from_binary(bin_img: np.ndarray) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    c = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4 and cv2.isContourConvex(approx):
        return approx.reshape(4, 2).astype(np.float32)
    rect = cv2.minAreaRect(c)
    return cv2.boxPoints(rect).astype(np.float32)


def _candidates_from_edges(img: np.ndarray) -> List[np.ndarray]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variants = [(0.10, 1), (0.33, 1), (0.55, 1)]
    quads = []
    for sigma, dil in variants:
        lo, hi = _auto_canny_thresholds(gray, sigma)
        edges = cv2.Canny(gray, lo, hi)
        if dil > 0:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.dilate(edges, k, iterations=dil)
        q = _contour_from_binary(edges)
        if q is not None:
            quads.append(q)
    return quads


def _candidates_from_binary(img: np.ndarray) -> List[np.ndarray]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    quads = []

    # Otsu
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    quads.append(_contour_from_binary(thr))

    # Adaptive mean & gaussian (her ikisi de belge senaryosunda iyi çalışır)
    for t in [cv2.ADAPTIVE_THRESH_MEAN_C, cv2.ADAPTIVE_THRESH_GAUSSIAN_C]:
        for C in [3, 7, 11]:
            block = max(21, (min(h, w)//25) | 1)
            ad = cv2.adaptiveThreshold(gray, 255, t, cv2.THRESH_BINARY, block, C)
            quads.append(_contour_from_binary(ad))
            quads.append(_contour_from_binary(255 - ad))

    # Flatten + Otsu
    flat = _background_flatten(gray, max(15, (min(h, w)//12) | 1))
    _, thr2 = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    quads.append(_contour_from_binary(thr2))
    quads.append(_contour_from_binary(255 - thr2))

    return [q for q in quads if q is not None]


def _candidates_from_gradient(img: np.ndarray) -> List[np.ndarray]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    soby = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    mag = cv2.convertScaleAbs(cv2.magnitude(sobx.astype(np.float32), soby.astype(np.float32)))
    mag = cv2.GaussianBlur(mag, (5, 5), 0)
    _, thr = cv2.threshold(mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, k, iterations=1)
    q = _contour_from_binary(thr)
    return [q] if q is not None else []


def _candidates_from_mser(img: np.ndarray) -> List[np.ndarray]:
    # MSER: metin ve kontrast farkı olan yüzeylerde belge bölgesini yakalamada faydalı
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mser = cv2.MSER_create(_delta=5, _min_area=200, _max_area=int(0.9 * gray.size))
    regions, _ = mser.detectRegions(gray)
    if not regions:
        return []
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions if p.shape[0] >= 3]
    if not hulls:
        return []
    # en büyük birkaç hull'u dene
    hulls = sorted(hulls, key=cv2.contourArea, reverse=True)[:8]
    quads = []
    for hcnt in hulls:
        peri = cv2.arcLength(hcnt, True)
        approx = cv2.approxPolyDP(hcnt, 0.02 * peri, True)
        if len(approx) >= 4:
            rect = cv2.minAreaRect(approx)
            box = cv2.boxPoints(rect).astype(np.float32)
            quads.append(box)
    return quads


def _candidates_from_hough(img: np.ndarray) -> List[np.ndarray]:
    # Klasik HoughLines yaklaşımı (dikdörtgen tahmini)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lo, hi = _auto_canny_thresholds(gray, 0.33)
    edges = cv2.Canny(gray, lo, hi)
    lines = cv2.HoughLines(edges, 1, np.pi/180, max(80, int(min(img.shape[:2])/5)))
    if lines is None:
        return []
    h_lines, v_lines = [], []
    for l in lines:
        rho, theta = l[0]
        if theta < np.pi/4 or theta > 3*np.pi/4:
            v_lines.append((rho, theta))
        else:
            h_lines.append((rho, theta))
    if len(h_lines) < 2 or len(v_lines) < 2:
        return []
    h_lines.sort(key=lambda x: x[0])
    v_lines.sort(key=lambda x: x[0])

    def intersect(l1, l2):
        (r1, t1), (r2, t2) = l1, l2
        A = np.array([[np.cos(t1), np.sin(t1)], [np.cos(t2), np.sin(t2)]])
        b = np.array([[r1], [r2]])
        try:
            x0, y0 = np.linalg.solve(A, b)
            return [float(x0[0]), float(y0[0])]
        except np.linalg.LinAlgError:
            return None

    pts = [
        intersect(h_lines[0], v_lines[0]),
        intersect(h_lines[0], v_lines[-1]),
        intersect(h_lines[-1], v_lines[-1]),
        intersect(h_lines[-1], v_lines[0])
    ]
    if any(p is None for p in pts):
        return []
    return [np.array(pts, dtype=np.float32)]


# =============================================================================
# Ana Bileşen
# =============================================================================
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
        if img is None or img.size == 0:
            raise ValueError("Input image is empty or None.")
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def _collect_all_candidates(self, img: np.ndarray) -> List[Tuple[np.ndarray, float, str, dict]]:
        """Birçok varyant ve aday üretici ile quad toplar ve puanlar."""
        cands: List[Tuple[np.ndarray, float, str, dict]] = []
        variants = _preprocess_variants(img)

        for vname, vimg in variants:
            # 1) Edge tabanlı
            for q in _candidates_from_edges(vimg):
                ok, m = _verify_quad(q, vimg.shape)
                sc = _score_candidate(q, vimg.shape)
                cands.append((q, sc, f"edges:{vname}", m))

            # 2) Binary tabanlı
            for q in _candidates_from_binary(vimg):
                ok, m = _verify_quad(q, vimg.shape)
                sc = _score_candidate(q, vimg.shape)
                cands.append((q, sc, f"binary:{vname}", m))

            # 3) Gradient (Sobel/Laplacian kombinasyonu)
            for q in _candidates_from_gradient(vimg):
                ok, m = _verify_quad(q, vimg.shape)
                sc = _score_candidate(q, vimg.shape)
                cands.append((q, sc, f"gradient:{vname}", m))

            # 4) MSER tabanlı
            for q in _candidates_from_mser(vimg):
                ok, m = _verify_quad(q, vimg.shape)
                sc = _score_candidate(q, vimg.shape)
                cands.append((q, sc, f"mser:{vname}", m))

            # 5) Hough tabanlı
            for q in _candidates_from_hough(vimg):
                ok, m = _verify_quad(q, vimg.shape)
                sc = _score_candidate(q, vimg.shape)
                cands.append((q, sc, f"hough:{vname}", m))

        # benzer quad'ları birleştirmek için küçük bir NMS: aynı köşeye çok yakınları ele
        merged: List[Tuple[np.ndarray, float, str, dict]] = []
        for q, sc, name, m in sorted(cands, key=lambda x: x[1], reverse=True):
            keep = True
            for mq, msc, _, _ in merged:
                if _quads_close(q, mq, tol=15.0):
                    keep = False
                    break
            if keep:
                merged.append((q, sc, name, m))

        return merged

    def _fallback_min_area_rect(self, img: np.ndarray) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lo, hi = _auto_canny_thresholds(gray, 0.33)
        edges = cv2.Canny(gray, lo, hi)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, k, iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        return cv2.boxPoints(rect).astype(np.float32)

    def run(self):
        img_obj = Image.get_frame(img=self.image, redis_db=self.redis_db)
        if img_obj is None or img_obj.value is None:
            raise ValueError("No input image provided or failed to load.")
        src = self._prepare_image(img_obj.value)

        # normalize boyut
        resized, scale = _resize_long_edge(src, 1800)

        # tüm adayları topla
        print("Adaylar toplanıyor (çoklu varyant + çoklu yöntem)...")
        candidates = self._collect_all_candidates(resized)

        best = None
        if candidates:
            best = max(candidates, key=lambda x: x[1])
            print(f"En iyi aday: {best[2]} | skor={best[1]:.3f} | metrics={best[3]}")
        else:
            print("Stratejilerden aday çıkmadı, algoritmik fallback deneniyor.")
            fb = self._fallback_min_area_rect(resized)
            if fb is not None:
                ok, m = _verify_quad(fb, resized.shape)
                sc = _score_candidate(fb, resized.shape)
                best = (fb, sc, "fallback:minAreaRect", m)
                print(f"Fallback üretildi: skor={sc:.3f}")

        if best is not None:
            quad_small = best[0]
            inv = 1.0 / scale
            quad_orig = (quad_small * inv).astype(np.float32)
            warped = _four_point_transform(src, quad_orig)
            if warped is None:
                print("Warp başarısız, tüm görüntü döndürülüyor.")
                warped = src
                quad_orig = np.array([[0, 0], [src.shape[1]-1, 0], [src.shape[1]-1, src.shape[0]-1], [0, src.shape[0]-1]], dtype=np.float32)
            chosen_name = best[2]
            chosen_score = float(best[1])
            chosen_metrics = best[3]
        else:
            print("Hiçbir yöntem başarılı olmadı, tüm görüntü döndürülüyor.")
            warped = src
            quad_orig = np.array([[0, 0], [src.shape[1]-1, 0], [src.shape[1]-1, src.shape[0]-1], [0, src.shape[0]-1]], dtype=np.float32)
            chosen_name = "full-image"
            chosen_score = 0.0
            chosen_metrics = {}

        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)

        self.context["src_quad"] = quad_orig.tolist()
        self.context["output_size"] = [warped.shape[1], warped.shape[0]]
        self.context["chosen_strategy"] = chosen_name
        self.context["chosen_score"] = chosen_score
        self.context["chosen_metrics"] = chosen_metrics

        return build_response(context=self)


# -----------------------------------------------------------------------------
# Küçük NMS benzeri: benzer quads yakınsa tekilleştir
# -----------------------------------------------------------------------------
def _quads_close(q1: np.ndarray, q2: np.ndarray, tol: float = 12.0) -> bool:
    a = _order_points(q1)
    b = _order_points(q2)
    d = np.linalg.norm(a - b, axis=1)
    return bool(np.all(d <= tol))


# -----------------------------------------------------------------------------
# Çalıştırıcı
# -----------------------------------------------------------------------------
Executor(sys.argv[1]).run()
