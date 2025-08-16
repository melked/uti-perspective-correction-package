import os
import sys
import cv2
import numpy as np
from typing import List, Tuple, Optional
from itertools import combinations


sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))
from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveCorrection.src.utils.response import build_response
from components.PerspectiveCorrection.src.models.PackageModel import PackageModel

# ----- YEREL TEST İÇİN SAHTE SINIFLAR (Yukarıdaki importlar yoksa) ----- #
class Image:
    def __init__(self, value): self.value = value

    @staticmethod
    def get_frame(img, redis_db): return Image(img) if not isinstance(img, Image) else img

    @staticmethod
    def set_frame(img, package_uID, redis_db): return img


class Component:
    def __init__(self, request,
                 bootstrap): self.request, self.bootstrap = request, bootstrap; self.uID = "test_uid"; self.redis_db = None


class Executor:
    def __init__(self, arg): print(f"Executor initialized with: {arg}")

    def run(self): print("Executor run called.")


def build_response(context): return {"context": context.context, "image": context.image}


class PackageModel:
    def __init__(self, **kwargs): self.__dict__.update(kwargs)


class Request:
    def __init__(self, data, params): self.data, self.params = data, params; self.model = None

    def get_param(self, key): return self.params.get(key)


# ----------------------------------------------------------------- #


# ---------------------- Yapılandırma ---------------------- #

class Config:
    """Bileşenin davranışını kontrol eden parametreleri merkezileştirir."""
    DEBUG_MODE = True  # True ise hata ayıklama görselleri üretir
    DEBUG_IMAGE_UID = "perspective_correction_debug"

    # Aday arama parametreleri
    CONTOUR_MIN_AREA_RATIO = 0.02
    CONTOUR_MAX_AREA_RATIO = 0.95
    CONTOUR_APPROX_EPSILON = 0.02
    HOUGH_THRESHOLD = 60
    HOUGH_MIN_LINE_LENGTH_RATIO = 0.05
    HOUGH_MAX_LINE_GAP = 15
    CORNER_MAX_CORNERS = 100
    CORNER_QUALITY_LEVEL = 0.01
    CORNER_MIN_DISTANCE = 20

    # Skorlama ağırlıkları (toplamları ~1.0 olmalı)
    W_AREA = 0.20
    W_CONVEXITY = 0.15
    W_ANGLE = 0.15
    W_SHAPE = 0.10
    W_ASPECT_RATIO = 0.10
    W_EDGE_DENSITY = 0.10
    W_PARALLELISM = 0.10
    W_SHARPNESS = 0.10  # Yeni skor: Netlik

    # Son işleme
    ENABLE_POST_ROTATION = True


# ---------------------- Geometri Yardımcıları ---------------------- #

def _order_points(pts: np.ndarray) -> np.ndarray:
    """Köşeleri tl, tr, br, bl sırasına sokar."""
    pts = pts.reshape(4, 2).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Verilen 4 noktaya göre perspektif düzeltmesi uygular."""
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(round(max(widthA, widthB)))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(round(max(heightA, heightB)))
    maxWidth, maxHeight = max(2, maxWidth), max(2, maxHeight)
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LANCZOS4)
    return warped


def _full_image_quad(image: np.ndarray) -> np.ndarray:
    """Görüntünün tamamını dörtgen olarak döndürür."""
    h, w = image.shape[:2]
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)


def _min_area_rect_quad(binary_or_gray: np.ndarray, ref_image: np.ndarray) -> np.ndarray:
    """En büyük konturun minimum alanlı dikdörtgenini bulur."""
    gray = cv2.cvtColor(binary_or_gray, cv2.COLOR_BGR2GRAY) if len(binary_or_gray.shape) == 3 else binary_or_gray.copy()
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return _full_image_quad(ref_image)
    c = max(contours, key=cv2.contourArea)
    return cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32)


# ---------------------- Ön İşleme Boru Hatları ---------------------- #

def _pre_soft(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 11)
    th = cv2.medianBlur(th, 5)
    return cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)


def _pre_medium(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    edges = cv2.Canny(eq, 50, 150, L2gradient=True)
    return cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)


def _pre_hard(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    unsharp = cv2.addWeighted(gray, 1.7, blur, -0.7, 0)
    edges = cv2.Canny(unsharp, 30, 100, L2gradient=True)
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)


def _pre_robust(gray: np.ndarray) -> np.ndarray:
    """YENİ: Gürültülü görüntüler için kenar korumalı filtreleme."""
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(denoised, 40, 120, L2gradient=True)
    return cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)


def _generate_edge_maps(image_bgr: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """Tüm ön işleme boru hatlarını çalıştırır ve etiketli kenar haritaları üretir."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return [
        ("soft", _pre_soft(gray)),
        ("medium", _pre_medium(gray)),
        ("hard", _pre_hard(gray)),
        ("robust", _pre_robust(gray)),  # Yeni boru hattı
    ]


# ---------------------- Aday Üretme ---------------------- #

def _find_quads_from_contours(binary_img: np.ndarray, ref_image: np.ndarray, config: Config) -> List[np.ndarray]:
    """Konturlardan dörtgen adayları bulur."""
    if binary_img is None or binary_img.size == 0: return []
    if len(binary_img.shape) == 3: binary_img = cv2.cvtColor(binary_img, cv2.COLOR_BGR2GRAY)
    if binary_img.dtype != np.uint8: binary_img = cv2.normalize(binary_img, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8)
    contours, _ = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return []

    H, W = ref_image.shape[:2]
    min_area = H * W * config.CONTOUR_MIN_AREA_RATIO
    max_area = H * W * config.CONTOUR_MAX_AREA_RATIO
    candidates = []
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]  # En büyük 20 konturu işle

    for c in contours:
        if not min_area < cv2.contourArea(c) < max_area: continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, config.CONTOUR_APPROX_EPSILON * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            candidates.append(approx.reshape(4, 2).astype(np.float32))
    return candidates


def _find_quads_from_hough(edges: np.ndarray, ref_image: np.ndarray, config: Config) -> List[np.ndarray]:
    """Hough dönüşümü ile bulunan çizgilerden dörtgen adayları türetir."""
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold=config.HOUGH_THRESHOLD,
                            minLineLength=max(30, int(config.HOUGH_MIN_LINE_LENGTH_RATIO * min(edges.shape[:2]))),
                            maxLineGap=config.HOUGH_MAX_LINE_GAP)
    if lines is None: return []

    def intersect(l1, l2) -> Optional[Tuple[float, float]]:
        x1, y1, x2, y2 = l1[0]
        x3, y3, x4, y4 = l2[0]
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if den == 0: return None
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den
        if 0 < t < 1 and 0 < u < 1:  # Sadece segmentler kesişiyorsa
            return float(x1 + t * (x2 - x1)), float(y1 + t * (y2 - y1))
        return None

    intersections = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            pt = intersect(lines[i], lines[j])
            if pt is not None: intersections.append(pt)

    if len(intersections) < 4: return []
    pts = np.array(intersections, dtype=np.float32)
    try:
        hull = cv2.convexHull(pts)
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, config.CONTOUR_APPROX_EPSILON * peri, True)
        if len(approx) == 4:
            return [approx.reshape(4, 2).astype(np.float32)]
    except Exception:
        return []  # convexHull hatası
    return []


def _find_quads_from_corners(gray_image: np.ndarray, config: Config) -> List[np.ndarray]:
    """YENİ: Shi-Tomasi ile bulunan güçlü köşelerden dörtgen adayları üretir."""
    corners = cv2.goodFeaturesToTrack(
        gray_image,
        maxCorners=config.CORNER_MAX_CORNERS,
        qualityLevel=config.CORNER_QUALITY_LEVEL,
        minDistance=config.CORNER_MIN_DISTANCE
    )
    if corners is None or len(corners) < 4:
        return []

    candidates = []
    pts = corners.reshape(-1, 2)
    # En geniş alanı kaplayan 4 köşe kombinasyonunu bul
    if len(pts) > 15:  # Çok fazlaysa hız için rastgele alt küme al
        indices = np.random.choice(len(pts), 15, replace=False)
        pts = pts[indices]

    max_area = 0
    best_quad = None
    for quad_indices in combinations(range(len(pts)), 4):
        quad = pts[list(quad_indices)]
        area = cv2.contourArea(quad)
        if area > max_area and cv2.isContourConvex(quad):
            max_area = area
            best_quad = quad
    if best_quad is not None:
        candidates.append(best_quad.astype(np.float32))
    return candidates


# ---------------------- Skorlama ---------------------- #

def _angle_score(quad: np.ndarray) -> float:
    q = _order_points(quad)

    def angle(p1, p2, p3):
        v1, v2 = p1 - p2, p3 - p2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    angles = [angle(q[i - 1], q[i], q[(i + 1) % 4]) for i in range(4)]
    return float(np.mean([1 - abs(a - 90) / 90 for a in angles]))


def _convexity_score(quad: np.ndarray) -> float:
    return 1.0 if cv2.isContourConvex(quad.astype(np.float32)) else 0.0


def _area_score(quad: np.ndarray, image: np.ndarray) -> float:
    area = abs(cv2.contourArea(quad))
    H, W = image.shape[:2]
    return float(np.clip(area / (H * W * 0.7), 0, 1))  # Alanın %70'ini hedefle


def _aspect_ratio_score(quad: np.ndarray) -> float:
    rect = _order_points(quad)
    (tl, tr, br, bl) = rect
    w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    h = (np.linalg.norm(tr - br) + np.linalg.norm(tl - bl)) / 2
    if min(w, h) < 1e-6: return 0.0
    ratio = max(w, h) / min(w, h)
    return float(max(0, 1 - abs(ratio - 1.41) / 1.41))  # A4 (sqrt(2)) oranını hedefle


def _edge_density_score(quad: np.ndarray, edges: np.ndarray) -> float:
    mask = np.zeros(edges.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [quad.astype(np.int32)], 255)
    inside_pixels = np.count_nonzero(mask)
    if inside_pixels == 0: return 0.0
    inside_edges = np.count_nonzero(cv2.bitwise_and(edges, edges, mask=mask))
    density = inside_edges / inside_pixels
    return float(np.clip(density * 10, 0, 1))  # Yoğunluğu ölçekle


def _parallelism_score(quad: np.ndarray) -> float:
    q = _order_points(quad)
    v_top, v_bottom = q[1] - q[0], q[2] - q[3]
    v_right, v_left = q[2] - q[1], q[3] - q[0]
    cos_sim1 = abs(np.dot(v_top, v_bottom) / (np.linalg.norm(v_top) * np.linalg.norm(v_bottom) + 1e-6))
    cos_sim2 = abs(np.dot(v_right, v_left) / (np.linalg.norm(v_right) * np.linalg.norm(v_left) + 1e-6))
    return float((cos_sim1 + cos_sim2) / 2.0)


def _sharpness_score(quad: np.ndarray, image: np.ndarray) -> float:
    """YENİ: Dörtgenin içindeki bölgenin netliğini (odak) ölçer."""
    try:
        warped = _four_point_transform(image, quad)
        if warped.size < 100: return 0.0  # Çok küçükse hesaplama
        gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray_warped, cv2.CV_64F).var()
        return float(np.clip(laplacian_var / 500, 0, 1.0))  # 500'e kadar olan varyansı normalize et
    except:
        return 0.0


def _score_quad(quad: np.ndarray, image: np.ndarray, edges: np.ndarray, config: Config) -> float:
    """Tüm metrikleri kullanarak bir dörtgen adayını skorlar."""
    return (config.W_AREA * _area_score(quad, image) +
            config.W_CONVEXITY * _convexity_score(quad) +
            config.W_ANGLE * _angle_score(quad) +
            config.W_SHAPE * 1.0 +  # _shape_score kaldırıldı, paralellik ve açı yeterli
            config.W_ASPECT_RATIO * _aspect_ratio_score(quad) +
            config.W_EDGE_DENSITY * _edge_density_score(quad, edges) +
            config.W_PARALLELISM * _parallelism_score(quad) +
            config.W_SHARPNESS * _sharpness_score(quad, image))


# ---------------------- Seçim ve Son İşleme ---------------------- #

def _select_best_quad(image: np.ndarray, candidates_by_source: dict, config: Config) -> Optional[np.ndarray]:
    """Tüm kaynaklardan gelen adaylar arasından en iyi skora sahip olanı seçer."""
    if not any(candidates_by_source.values()):
        return None

    scored = []
    for source, data in candidates_by_source.items():
        candidates = data["candidates"]
        edges = data["edges"]
        for q in candidates:
            s = _score_quad(q, image, edges, config)
            scored.append({"quad": q, "score": s, "source": source})

    if not scored: return None

    best = max(scored, key=lambda x: x["score"])
    print(f"Best quad found by '{best['source']}' with score: {best['score']:.3f}")  # Bilgilendirme
    return best["quad"]


def _post_process(image: np.ndarray, config: Config) -> np.ndarray:
    """YENİ: İsteğe bağlı son işleme adımları uygular."""
    if config.ENABLE_POST_ROTATION:
        h, w = image.shape[:2]
        if w > h:  # Eğer genişlik yükseklikten fazlaysa, dikey hale getir
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image


# ---------------------- Hata Ayıklama ---------------------- #

def _draw_debug_info(src: np.ndarray, edge_maps: List[Tuple[str, np.ndarray]], candidates: dict,
                     best_quad: np.ndarray) -> np.ndarray:
    """YENİ: Hata ayıklama için görsel bir çıktı üretir."""
    h, w = src.shape[:2]
    # 4 kenar haritasını birleştirmek için tuval
    edge_canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)
    positions = [(0, 0), (0, w), (h // 2, 0), (h // 2, w)]  # 2x2 gridde daha mantıklı
    if h * 2 > 2000 or w * 2 > 2000:  # Tuvali küçült
        edge_canvas = np.zeros((h, w, 3), dtype=np.uint8)
        positions = [(0, 0), (0, w // 2), (h // 2, 0), (h // 2, w // 2)]

    for i, (name, emap) in enumerate(edge_maps):
        if i >= 4: break
        resized_emap = cv2.resize(emap, (w // 2, h // 2))
        emap_color = cv2.cvtColor(resized_emap, cv2.COLOR_GRAY2BGR)
        cv2.putText(emap_color, name, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset, x_offset = positions[i]
        edge_canvas[y_offset:y_offset + h // 2, x_offset:x_offset + w // 2] = emap_color

    # Ana görsel üzerine adayları ve en iyiyi çiz
    out_img = src.copy()
    # Tüm adayları ince bir renkle çiz
    colors = {"contour": (255, 0, 0), "hough": (0, 255, 0), "corner": (0, 0, 255)}
    for source, data in candidates.items():
        for q in data["candidates"]:
            cv2.polylines(out_img, [q.astype(np.int32)], True, colors.get(source, (128, 128, 128)), 1)
    # En iyi dörtgeni kalın çiz
    cv2.polylines(out_img, [best_quad.astype(np.int32)], True, (0, 255, 255), 3, cv2.LINE_AA)

    # İki görseli birleştir
    final_h, final_w = max(out_img.shape[0], edge_canvas.shape[0]), out_img.shape[1] + edge_canvas.shape[1]
    debug_canvas = np.zeros((final_h, final_w, 3), dtype=np.uint8)
    debug_canvas[:out_img.shape[0], :out_img.shape[1]] = out_img
    debug_canvas[:edge_canvas.shape[0], out_img.shape[1]:] = edge_canvas

    return cv2.resize(debug_canvas, (1920, int(1920 * final_h / final_w)))  # Boyut standardizasyonu


# ---------------------- Component Sınıfı ---------------------- #

class PerspectiveCorrection(Component):
    """
    - Çoklu strateji ve gelişmiş skorlama ile her arka planda sağlam.
    - Yapılandırılabilir parametreler ile esnek.
    - Otomatik döndürme ve hata ayıklama modu gibi ek yetenekler.
    """

    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.context = {}
        # Bootstrap'ten gelen config'i al, request'te varsa ez
        self.config = Config()
        if bootstrap and 'config' in bootstrap:
            for key, value in bootstrap['config'].items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        """Bileşen başlatılırken varsayılan yapılandırmayı ayarlar."""
        return {"config": {k: v for k, v in Config.__dict__.items() if not k.startswith('__')}}

    @staticmethod
    def _prepare_image(img: np.ndarray) -> np.ndarray:
        if img is None or img.size == 0: raise ValueError("Input image is empty")
        if img.dtype != np.uint8: img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def run(self):
        img_obj = Image.get_frame(img=self.image, redis_db=self.redis_db)
        if img_obj is None or img_obj.value is None: raise ValueError("Failed to load input image")
        src = self._prepare_image(img_obj.value)

        # 1. Kenar haritalarını ve temel gri görüntüyü üret
        edge_maps = _generate_edge_maps(src)
        gray_src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

        # 2. Tüm yöntemlerle adayları topla
        candidates_by_source = {}
        for name, emap in edge_maps:
            key = f"contour_{name}"
            candidates_by_source[key] = {
                "candidates": _find_quads_from_contours(emap, src, self.config),
                "edges": emap
            }
            key = f"hough_{name}"
            candidates_by_source[key] = {
                "candidates": _find_quads_from_hough(emap, src, self.config),
                "edges": emap
            }

        candidates_by_source["corner_gray"] = {
            "candidates": _find_quads_from_corners(gray_src, self.config),
            "edges": edge_maps[1][1]  # Temsili olarak medium edge map kullan
        }

        # 3. En iyi adayı seç
        best_quad = _select_best_quad(src, candidates_by_source, self.config)

        # 4. Fallback mekanizması
        if best_quad is None:
            print("No suitable quad found. Falling back to minAreaRect.")
            best_quad = _min_area_rect_quad(edge_maps[0][1], src)

        # 5. Düzeltme ve son işleme
        warped = _four_point_transform(src, best_quad)
        warped = _post_process(warped, self.config)

        # 6. Sonuçları hazırla ve hata ayıklama görseli oluştur
        if self.config.DEBUG_MODE:
            # Flatten candidates for debug drawing
            flat_candidates = {}
            for k, v in candidates_by_source.items():
                source_type = k.split('_')[0]
                if source_type not in flat_candidates: flat_candidates[source_type] = {"candidates": []}
                flat_candidates[source_type]["candidates"].extend(v["candidates"])

            debug_img = _draw_debug_info(src, edge_maps, flat_candidates, best_quad)
            debug_img_obj = Image(value=debug_img)
            Image.set_frame(img=debug_img_obj, package_uID=self.config.DEBUG_IMAGE_UID, redis_db=self.redis_db)

        # 7. Çıktıyı ayarla
        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)
        self.context["src_quad"] = best_quad.tolist()
        self.context["output_size"] = [warped.shape[1], warped.shape[0]]
        self.context["method"] = {
            "pipelines": [name for name, _ in edge_maps],
            "finders": ["contour", "hough", "corner"],
            "fallbacks": ["min_area_rect", "full_image_if_needed"],
            "debug_image_uid": self.config.DEBUG_IMAGE_UID if self.config.DEBUG_MODE else "disabled"
        }
        return build_response(context=self)


if __name__ == '__main__':
    # Bu blok, bileşenin dışında yerel test için kullanılır
    print("Running PerspectiveCorrection in local test mode.")
    # Örnek bir görüntü yükle (dosya yolunu kendi sisteminize göre değiştirin)
    try:
        # test_image = cv2.imread("path/to/your/test/image.jpg")
        # Örnek olarak eğimli bir A4 kağıdı oluşturalım
        h, w = 600, 800
        test_image = np.full((h, w, 3), (150, 160, 170), dtype=np.uint8)
        pts = np.array([[150, 100], [w - 200, 50], [w - 50, h - 80], [80, h - 120]])
        cv2.fillPoly(test_image, [pts], (255, 255, 255))
        cv2.putText(test_image, "TEST DOCUMENT", (200, h // 2), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 2)
        cv2.polylines(test_image, [pts], True, (0, 0, 0), 3)

        if test_image is None:
            raise FileNotFoundError("Test image not found.")

        # Sahte request ve bootstrap objeleri oluştur
        mock_request = Request(data={}, params={"inputImage": test_image})
        mock_bootstrap = PerspectiveCorrection.bootstrap({})

        # Bileşeni çalıştır
        component = PerspectiveCorrection(mock_request, mock_bootstrap)
        result = component.run()

        # Sonuçları göster
        output_image = result["image"].value
        cv2.imshow("Original Image", cv2.resize(test_image, (output_image.shape[1], output_image.shape[0])))
        cv2.imshow("Corrected Image", output_image)

        if Config.DEBUG_MODE:
            # Gerçek bir sistemde bu redis'ten okunurdu
            debug_img_obj = Image.get_frame(img=Config.DEBUG_IMAGE_UID, redis_db=None)
            # Yerel testte, debug fonksiyonunu tekrar çağırıp sonucu alıyoruz (basitleştirme)
            # Gerçekte debug görüntüsü zaten set_frame ile kaydedilmiş olurdu.
            # Bu yüzden bu kısım sadece görselleştirme için.
            if 'debug_image_uid' in result['context']['method']:
                print("\nDebug image would be available under UID:", result['context']['method']['debug_image_uid'])
                # Hata ayıklama görüntüsünü göstermek için, onu yeniden oluşturmamız gerekir (test senaryosu)
                gray_src = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
                edge_maps = _generate_edge_maps(test_image)
                candidates_by_source = {
                    "contour": {"candidates": _find_quads_from_contours(edge_maps[0][1], test_image, Config())},
                    "hough": {"candidates": _find_quads_from_hough(edge_maps[1][1], test_image, Config())},
                    "corner": {"candidates": _find_quads_from_corners(gray_src, Config())}
                }
                debug_viz = _draw_debug_info(test_image, edge_maps, candidates_by_source,
                                             np.array(result['context']['src_quad']))
                cv2.imshow("Debug Visualization", debug_viz)

        print("\nContext:")
        import json

        print(json.dumps(result["context"], indent=2))

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred: {e}")

    # Normalde bu satır en sonda olurdu
Executor(sys.argv[1]).run()