
import os
import sys
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

try:
    from sdks.novavision.src.media.image import Image
    from sdks.novavision.src.base.component import Component
    from sdks.novavision.src.helper.executor import Executor
    from components.PerspectiveCorrection.src.utils.response import build_response
    from components.PerspectiveCorrection.src.models.PackageModel import PackageModel
except ImportError:
    print("UYARI: SDK sınıfları bulunamadı, yerel test için sahte sınıflar kullanılıyor.")


    class Image:
        def __init__(self, value): self.value = value

        @staticmethod
        def get_frame(img, redis_db): return Image(img)

        @staticmethod
        def set_frame(img, package_uID, redis_db): return img


    class Component:
        def __init__(self, request, bootstrap): self.request, self.uID = request, "test_uid"


    def build_response(context):
        return context.context


# ----------------------------------------------------------------------


def _order_points(pts: np.ndarray) -> np.ndarray:
    # ... (Bu fonksiyon aynı kalıyor) ...
    pts = pts.reshape(4, 2).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
    return rect


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    # ... (Bu fonksiyon aynı kalıyor) ...
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect
    width = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
    height = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))
    maxWidth, maxHeight = int(round(width)), int(round(height))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LANCZOS4)


def _full_image_quad(image: np.ndarray) -> np.ndarray:
    # ... (Bu fonksiyon aynı kalıyor) ...
    h, w = image.shape[:2]
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)


# --- YENİ EKLENEN PİPELİNELAR ---
# Mevcut pipeline'larınıza dokunmadan, daha zorlu durumlar için yenilerini ekliyoruz.
PIPELINE_CONFIGS: List[Dict[str, Any]] = [
    # --- YENİ: Tekstürlü arkaplanları ve gölgeleri yok etmek için ---
    {
        "name": "heavy_blur_otsu",
        "preprocess": {"type": "gaussian", "ksize": (15, 15)},  # Çok güçlü bulanıklaştırma
        "threshold": {"type": "otsu_inv"},
        "morphology": {"op": "close_open", "ksize": (15, 15)}
    },
    # --- YENİ: Renk ve parlaklık farkını en iyi ayıran LAB uzayı ---
    {
        "name": "lab_bright_achromatic",
        "color_space": "lab",
        "threshold": {"type": "lab_bright"},  # Sadece parlak ve renksiz alanları al
        "morphology": {"op": "close", "ksize": (9, 9), "iter": 2}
    },
    # --- Mevcut Pipeline'larınız ---
    {
        "name": "sharpen_adaptive",
        "preprocess": {"type": "bilateral", "d": 9, "sigma": 75},
        "threshold": {"type": "adaptive", "block": 11, "C": 2, "invert": True},
        "morphology": {"op": "close_open", "ksize": (5, 5)}
    },
    {
        "name": "clahe_canny",
        "preprocess": {"type": "clahe"},
        "threshold": {"type": "canny", "th1": 50, "th2": 150},
        "morphology": {"op": "close", "ksize": (5, 5)}
    },
    {
        "name": "bright_blur_canny",
        "preprocess": {"type": "gamma", "gamma": 1.8},
        "threshold": {"type": "canny", "th1": 30, "th2": 120},
        "morphology": {"op": "close", "ksize": (7, 7), "iter": 2}
    },
    {
        "name": "inverse_otsu",
        "preprocess": {"type": "gaussian", "ksize": (5, 5)},
        "threshold": {"type": "otsu_inv"},
        "morphology": {"op": "close_open", "ksize": (5, 5)}
    },
    {
        "name": "color_segment_hsv",
        "color_space": "hsv",
        "threshold": {"type": "inRange", "lower": [0, 0, 150], "upper": [180, 50, 255]},
        # Aralığı biraz daha hassas hale getirdik
        "morphology": {"op": "close_open", "ksize": (7, 7)}
    }
]


def _process_pipeline(image: np.ndarray, config: Dict[str, Any]) -> Optional[np.ndarray]:
    # Adım 1: Renk Uzayı Değişimi
    if config.get("color_space") == "hsv":
        proc_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif config.get("color_space") == "lab":  # YENİ
        proc_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    else:
        proc_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Adım 2: Ön İşleme
    # ... (Bu bölüm aynı kalıyor) ...
    preproc_conf = config.get("preprocess", {})
    if preproc_conf.get("type") == "bilateral":
        proc_img = cv2.bilateralFilter(proc_img, preproc_conf["d"], preproc_conf["sigma"], preproc_conf["sigma"])
    elif preproc_conf.get("type") == "gaussian":
        proc_img = cv2.GaussianBlur(proc_img, preproc_conf["ksize"], 0)
    elif preproc_conf.get("type") == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        proc_img = clahe.apply(proc_img)
    elif preproc_conf.get("type") == "gamma":
        table = np.array([((i / 255.0) ** preproc_conf["gamma"]) * 255 for i in np.arange(256)]).astype("uint8")
        proc_img_color = cv2.LUT(image, table)
        proc_img = cv2.cvtColor(proc_img_color, cv2.COLOR_BGR2GRAY)

    # Adım 3: Eşikleme veya Kenar Tespiti
    thresh_conf = config.get("threshold", {})
    binary_mask = None
    if thresh_conf.get("type") == "adaptive":
        method = cv2.THRESH_BINARY_INV if thresh_conf.get("invert") else cv2.THRESH_BINARY
        binary_mask = cv2.adaptiveThreshold(proc_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, method, thresh_conf["block"],
                                            thresh_conf["C"])
    elif thresh_conf.get("type") == "canny":
        binary_mask = cv2.Canny(proc_img, thresh_conf["th1"], thresh_conf["th2"])
    elif thresh_conf.get("type") == "otsu_inv":
        _, binary_mask = cv2.threshold(proc_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif thresh_conf.get("type") == "inRange":
        lower = np.array(thresh_conf["lower"], dtype="uint8")
        upper = np.array(thresh_conf["upper"], dtype="uint8")
        binary_mask = cv2.inRange(proc_img, lower, upper)
    # --- YENİ LAB EŞİKLEME ---
    elif thresh_conf.get("type") == "lab_bright":
        L, A, B = cv2.split(proc_img)
        # Sadece çok parlak (L > 160) ve renksiz (A ve B 128'e yakın) pikselleri al
        _, l_mask = cv2.threshold(L, 160, 255, cv2.THRESH_BINARY)
        a_mask = cv2.inRange(A, 118, 138)
        b_mask = cv2.inRange(B, 118, 138)
        binary_mask = cv2.bitwise_and(l_mask, cv2.bitwise_and(a_mask, b_mask))

    if binary_mask is None: return None

    # Adım 4: Morfolojik Operasyonlar
    # ... (Bu bölüm aynı kalıyor) ...
    morph_conf = config.get("morphology", {})
    if "ksize" in morph_conf:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_conf["ksize"])
        iters = morph_conf.get("iter", 1)
        if morph_conf["op"] == "close":
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=iters)
        elif morph_conf["op"] == "close_open":
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=iters)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=iters)

    # Adım 5: Konturlardan Dörtgen Bulma
    # ... (Bu bölüm aynı kalıyor) ...
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


# --- YENİ HOUGH DÖNÜŞÜMÜ STRATEJİSİ ---
def _pipeline_hough_vote_map(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Görüntüdeki düz çizgileri bularak bir 'oy haritası' oluşturur.
    Tekstürlü zeminlere karşı çok dayanıklıdır.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
    if lines is None:
        return None

    # Bulunan çizgileri boş bir maske üzerine çizerek oy haritası oluştur
    vote_map = np.zeros_like(edges)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(vote_map, (x1, y1), (x2, y2), 255, 3)

    # Çizgileri birleştirerek kapalı bir alan oluştur
    vote_map = cv2.morphologyEx(vote_map, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    # Bu haritadan en büyük dörtgeni bul
    contours, _ = cv2.findContours(vote_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None

    c = max(contours, key=cv2.contourArea)
    # En büyük konturun minimum alanlı dikdörtgenini alarak daha düzgün bir dörtgen elde et
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    return box.astype(np.float32)


def _score_quad(quad: np.ndarray, image_shape: Tuple[int, int]) -> float:
    # ... (Bu fonksiyon aynı kalıyor) ...
    area = abs(cv2.contourArea(quad))
    img_area = image_shape[0] * image_shape[1]
    area_score = 1.0 if 0.05 < area / img_area < 0.95 else 0.1

    ordered_quad = _order_points(quad)
    angles = []
    for i in range(4):
        p1, p2, p3 = ordered_quad[(i - 1) % 4], ordered_quad[i], ordered_quad[(i + 1) % 4]
        v1, v2 = p1 - p2, p3 - p2
        angle_dot = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(angle_dot, -1.0, 1.0)))
        angles.append(angle)
    angle_score = np.mean([1 - abs(a - 90) / 90 for a in angles])

    w = (np.linalg.norm(ordered_quad[0] - ordered_quad[1]) + np.linalg.norm(ordered_quad[2] - ordered_quad[3])) / 2
    h = (np.linalg.norm(ordered_quad[1] - ordered_quad[2]) + np.linalg.norm(ordered_quad[0] - ordered_quad[3])) / 2
    aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
    ratio_score = max(0, 1 - abs(aspect_ratio - 1.4) / 5.0)

    return (area_score * 0.4) + (angle_score * 0.4) + (ratio_score * 0.2)


class PerspectiveCorrection(Component):  # YENİ SINIF ADI
    def __init__(self, request, bootstrap):
        # ... (Bu bölüm aynı kalıyor, sınıf adını PerspectiveCorrection olarak düzelttim) ...
        super().__init__(request, bootstrap)
        self.context = {}
        # self.request.model = PackageModel(**(self.request.data)) # Bu satır yerel testte hata verebilir
        self.image = request.get_param("inputImage") if hasattr(request, 'get_param') else request.image

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def _prepare_image(self, img: np.ndarray) -> np.ndarray:
        # ... (Bu fonksiyon aynı kalıyor) ...
        if img is None or img.size == 0:
            raise ValueError("Input image is empty or None.")
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def _find_best_quad(self, image: np.ndarray) -> Tuple[np.ndarray, str, float]:
        """Tüm pipeline'ları ve stratejileri dener ve en yüksek skorlu dörtgeni bulur."""
        candidates = []

        # 1. Pipeline konfigürasyonlarını çalıştır
        for config in PIPELINE_CONFIGS:
            try:
                quad = _process_pipeline(image, config)
                if quad is not None:
                    score = _score_quad(quad, image.shape)
                    candidates.append({"quad": quad, "source": config["name"], "score": score})
            except Exception as e:
                print(f"Pipeline '{config['name']}' hata verdi: {e}")

        # 2. YENİ Hough stratejisini çalıştır
        try:
            quad = _pipeline_hough_vote_map(image)
            if quad is not None:
                score = _score_quad(quad, image.shape)
                candidates.append({"quad": quad, "source": "hough_vote_map", "score": score})
        except Exception as e:
            print(f"Pipeline 'hough_vote_map' hata verdi: {e}")

        if not candidates:
            return _full_image_quad(image), "fallback_full_image", 0.0

        best_candidate = max(candidates, key=lambda c: c["score"])
        return best_candidate["quad"], best_candidate["source"], best_candidate["score"]

    def run(self):
        # ... (Bu bölüm aynı kalıyor, sınıf adını PerspectiveCorrection olarak düzelttim) ...
        img_obj = Image.get_frame(img=self.image, redis_db=None)
        src_img = self._prepare_image(img_obj.value)

        best_quad, source, score = self._find_best_quad(src_img)
        print(f"En iyi aday '{source}' pipeline'ından {score:.2f} skorla bulundu.")

        warped = _four_point_transform(src_img, best_quad)

        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=None)
        self.context = {
            "src_quad": best_quad.tolist(),
            "output_size": [warped.shape[1], warped.shape[0]],
            "best_pipeline": source,
            "confidence_score": score
        }
        return self


# YEREL TEST İÇİN (Eğer isterseniz bu bloğu kullanabilirsiniz)
if __name__ == '__main__':
    # Test etmek istediğiniz görselin yolunu buraya yazın
    image_path = 'IMG_6167.jpg'  # veya 'IMG_6166.jpg'

    if not os.path.exists(image_path):
        print(f"HATA: '{image_path}' bulunamadı. Lütfen dosya yolunu kontrol edin.")
    else:
        # Sahte request nesnesi oluştur
        class MockRequest:
            def __init__(self, image_path):
                self.image = cv2.imread(image_path)

            def get_param(self, key):
                if key == "inputImage":
                    return self.image


        mock_request = MockRequest(image_path)

        # Bileşeni çalıştır
        component = PerspectiveCorrection(mock_request, {})
        result_component = component.run()

        # Sonucu al ve göster
        corrected_image = result_component.image.value

        # Görselleri yan yana koyarak karşılaştır
        original_image = cv2.imread(image_path)
        h, w, _ = original_image.shape
        corrected_image_resized = cv2.resize(corrected_image, (int(w / 2), int(h / 2)))
        original_image_resized = cv2.resize(original_image, (int(w / 2), int(h / 2)))

        # Çıktı dosyası
        output_path = 'result_comparison.jpg'
        cv2.imwrite(output_path, corrected_image)
        print(f"Sonuç '{output_path}' dosyasına kaydedildi.")

Executor(sys.argv[1]).run() 