import os
import sys
import cv2
import numpy as np
from typing import List, Tuple, Optional
from itertools import combinations

# SDK KISIMLARI OLDUĞU GİBİ KALIYOR
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))
from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveCorrection.src.utils.response import build_response
from components.PerspectiveCorrection.src.models.PackageModel import PackageModel


# ---------------------- 1. Preprocessing Pipelines (Geliştirildi) ---------------------- #
# Ön işleme adımlarınızdaki parametreleri ve yöntemleri daha sağlam hale getiriyoruz.

def _pre_soft(gray: np.ndarray) -> np.ndarray:
    """Desenli/karmaşık arka planlarda daha iyi sonuç için: adaptif eşik + morfoloji."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # THRESH_BINARY_INV, genellikle beyaz zemin üzerindeki nesneleri daha iyi ayırır.
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 21, 10)
    # Gürültüyü ve küçük delikleri kapatmak için daha güçlü bir kapama işlemi.
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    return th


def _pre_medium(gray: np.ndarray) -> np.ndarray:
    """Genel amaçlı: CLAHE + Canny. Parametreler optimize edildi."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    # Eşik aralığı daha genel durumları yakalamak için ayarlandı.
    edges = cv2.Canny(eq, 60, 180, L2gradient=True)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    return edges


def _pre_hard(gray: np.ndarray) -> np.ndarray:
    """Uzak/az kontrast: unsharp + düşük eşikli Canny. Daha belirgin kenarlar için."""
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    unsharp = cv2.addWeighted(gray, 1.8, blur, -0.8, 0)  # Keskinleştirme artırıldı.
    edges = cv2.Canny(unsharp, 40, 120, L2gradient=True)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
    return edges


def _pre_color_suppressed(gray: np.ndarray, bgr: np.ndarray) -> np.ndarray:
    """Renk bilgisini kullanarak kenarları ortaya çıkaran en güçlü pipeline."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel, _, _ = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_channel)
    grad = cv2.morphologyEx(l_eq, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    _, th = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Gürültüleri temizlemek ve dokümanı tek bir parça haline getirmek için kapama.
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    return th


def _generate_edge_maps(bgr: np.ndarray) -> List[np.ndarray]:
    """Tüm ön işleme stratejilerini uygulayarak kenar haritaları listesi oluşturur."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return [
        _pre_soft(gray.copy()),
        _pre_medium(gray.copy()),
        _pre_hard(gray.copy()),
        _pre_color_suppressed(gray.copy(), bgr)
    ]


# ---------------------- 2. Geometry Helpers (Mevcut haliyle iyi, korunuyor) ---------------------- #

def _order_points(pts: np.ndarray) -> np.ndarray:
    """Köşeleri tl, tr, br, bl sırasına sokar."""
    # Bu fonksiyonunuz zaten standart ve doğru çalışıyor. Değişikliğe gerek yok.
    pts = pts.reshape(4, 2).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Verilen 4 noktaya göre görüntüye perspektif dönüşümü uygular.
    Artık hem düzeltilmiş görüntüyü hem de hedef noktaları döndürüyor.
    """
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0], [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LANCZOS4)
    return warped, dst


# --- 3. Candidate Generation (Hough kırılgandı, Kontur'a odaklanıldı) ---------------------- #

def _find_quads_from_contours(binary_img: np.ndarray, ref_image: np.ndarray,
                              min_area_ratio: float = 0.05) -> List[np.ndarray]:
    """İkili görüntüdeki konturları analiz ederek dörtgen adayları bulur."""
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return []

    H, W = ref_image.shape[:2]
    img_area = H * W
    min_area = img_area * min_area_ratio

    candidates = []
    # En büyük 10 kontura bakmak genellikle yeterlidir ve performansı artırır.
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        if cv2.contourArea(c) < min_area: continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.025 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            candidates.append(approx.astype(np.float32))
    return candidates


# _find_quads_from_hough fonksiyonunuzu siliyoruz çünkü çok fazla özel duruma dayanıyor ve
# kırılgandı. Onun yerine 4 farklı ve güçlü kontur bulma yöntemine odaklanmak
# sistemi çok daha sağlam ("robust") yapar.

# ---------------------- 4. Scoring (Zayıf halka, tamamen yenilendi) ---------------------- #

def _score_quad(quad: np.ndarray) -> float:
    """Bir dörtgen adayının 'doküman olma' olasılığını geometrik özelliklere göre 0-1 arasında puanlar."""
    q = quad.reshape(4, 2)

    def get_angle(p1, p2, p3):
        v1, v2 = p1 - p2, p3 - p2
        cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    angles = [get_angle(q[i], q[(i + 1) % 4], q[(i + 2) % 4]) for i in range(4)]
    # 90 dereceden sapmayı cezalandıran Gaussian (yumuşak) bir skorlama.
    angle_score = np.mean([np.exp(-((a - 90) ** 2) / (2 * 10 ** 2)) for a in angles])

    rect = _order_points(q)
    w = (np.linalg.norm(rect[0] - rect[1]) + np.linalg.norm(rect[2] - rect[3])) / 2
    h = (np.linalg.norm(rect[1] - rect[2]) + np.linalg.norm(rect[0] - rect[3])) / 2
    aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
    # A4 (1.41) gibi oranlara yakınlığı ödüllendirir.
    aspect_score = np.exp(-((aspect_ratio - 1.4) ** 2) / (2 * 1.0 ** 2))

    # Geometrik doğruluk (açılar) en önemli kriter olduğu için ağırlığı daha yüksek.
    return 0.7 * angle_score + 0.3 * aspect_score


# ---------------------- 5. Selection (Güven skoru kontrolü eklendi) ---------------------- #

def _select_best_quad(candidates: List[np.ndarray]) -> Tuple[Optional[np.ndarray], float]:
    """Tüm adaylar arasından en yüksek skorlu olanı seçer."""
    if not candidates:
        return None, 0.0

    scored_candidates = [(c, _score_quad(c)) for c in candidates]
    best_quad, best_score = max(scored_candidates, key=lambda item: item[1])

    return best_quad, best_score


# ---------------------- 6. Yeni Yardımcı Fonksiyon: Görüntü İyileştirme ---------------------- #

def _enhance_image(image: np.ndarray) -> np.ndarray:
    """Düzeltilmiş görüntünün okunabilirliğini artırır."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    return cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)


# ---------------------- Component (Ana Mantık, Hedefe Göre Yeniden Yazıldı) ---------------------- #

class PerspectiveCorrection(Component):
    """
    - Sizin kod yapınızı koruyarak, her ortamda daha sağlam çalışacak şekilde güçlendirildi.
    - Düzeltilmiş ve iyileştirilmiş dokümanı, orijinal görüntünün üzerine yerleştirir.
    """
    MIN_CONFIDENCE_SCORE = 0.70  # Başarılı bir tespit için gereken minimum güven skoru

    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.context = {}
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    @staticmethod
    def _prepare_image(img: np.ndarray) -> np.ndarray:
        if img is None or img.size == 0:
            raise ValueError("Giriş görüntüsü boş veya geçersiz.")
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def run(self):
        img_obj = Image.get_frame(img=self.image, redis_db=self.redis_db)
        if img_obj is None or img_obj.value is None:
            raise ValueError("Giriş görüntüsü yüklenemedi.")

        src_original = self._prepare_image(img_obj.value)
        src = src_original.copy()

        # Adım 1: Tüm ön işleme yöntemleriyle kenar haritaları oluştur
        edge_maps = _generate_edge_maps(src)

        # Adım 2: Tüm haritalardan aday dörtgenleri topla
        all_candidates = []
        for emap in edge_maps:
            all_candidates.extend(_find_quads_from_contours(emap, src))

        # Adım 3: En iyi adayı ve skorunu seç
        best_quad, best_score = _select_best_quad(all_candidates)

        self.context["best_score_found"] = best_score

        # Adım 4: Güven Kontrolü - En önemli adımdır!
        # Eğer en iyi adayın skoru bile yeterince yüksek değilse, risk alma ve orijinal görüntüyü döndür.
        if best_score < self.MIN_CONFIDENCE_SCORE:
            self.context["status"] = f"Failure: No document found with confidence > {self.MIN_CONFIDENCE_SCORE}"
            self.context["src_quad"] = None
            final_image = src_original
        else:
            self.context["status"] = "Success: Document corrected and overlaid on the original image."
            self.context["src_quad"] = best_quad.tolist()

            # Adım 5: Düzeltme ve İyileştirme
            rect = _order_points(best_quad)
            warped_doc, dst_pts = _four_point_transform(src, rect)
            enhanced_doc = _enhance_image(warped_doc)

            # Adım 6: Orijinal Görüntü Üzerine Yerleştirme (Overlay)
            # Düzeltilmiş görüntüyü, orijinaldeki yerine geri "eğmek" için ters dönüşüm matrisi hesapla
            inverse_matrix = cv2.getPerspectiveTransform(dst_pts, rect)

            # İyileştirilmiş dokümanı orijinal görüntünün boyutlarına geri warp et
            h, w = src.shape[:2]
            inversely_warped = cv2.warpPerspective(enhanced_doc, inverse_matrix, (w, h))

            # Maskeleme ile iki görüntüyü birleştir
            mask = np.zeros_like(src, dtype=np.uint8)
            cv2.fillConvexPoly(mask, rect.astype(np.int32), (255, 255, 255))
            mask_inv = cv2.bitwise_not(mask)

            # Orijinal görüntüden doküman alanını çıkar
            background = cv2.bitwise_and(src_original, mask_inv)
            # Geri warp edilmiş dokümanı al
            foreground = cv2.bitwise_and(inversely_warped, mask)

            # İkisini birleştir
            final_image = cv2.add(background, foreground)

        # Sonuçları kaydet
        img_obj.value = final_image
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)
        self.context["output_size"] = [final_image.shape[1], final_image.shape[0]]
        self.context["method"] = {
            "pipelines_used": ["soft_adaptive", "medium_clahe_canny", "hard_unsharp_canny", "color_suppressed_lab"],
            "candidate_source": "contours_only",
            "confidence_threshold": self.MIN_CONFIDENCE_SCORE
        }
        return build_response(context=self)

# Executor(sys.argv[1]).run()