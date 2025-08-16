import os
import sys
import cv2
import numpy as np
from typing import List, Tuple, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveCorrection.src.utils.response import build_response
from components.PerspectiveCorrection.src.models.PackageModel import PackageModel


# ---------------------- 1. Preprocessing Pipelines (İyileştirildi) ---------------------- #

def _pre_soft(gray: np.ndarray) -> np.ndarray:
    """Karmaşık arka planlar için adaptif eşikleme ve morfolojik temizleme."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 21, 10)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    return th


def _pre_medium(gray: np.ndarray) -> np.ndarray:
    """Genel amaçlı, kontrastı artırılmış Canny kenar tespiti."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    edges = cv2.Canny(eq, 60, 180, L2gradient=True)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    return edges


def _pre_hard(gray: np.ndarray) -> np.ndarray:
    """Düşük kontrastlı veya uzak çekimler için keskinleştirme ve hassas Canny."""
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    unsharp = cv2.addWeighted(gray, 1.8, blur, -0.8, 0)
    edges = cv2.Canny(unsharp, 40, 120, L2gradient=True)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    return edges


def _pre_lab_color(bgr: np.ndarray) -> np.ndarray:
    """Renk bilgisini kullanarak kenarları vurgulayan en güçlü pipeline."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel, _, _ = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_channel)
    grad = cv2.morphologyEx(l_eq, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    _, th = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
    return th


def generate_edge_maps(bgr: np.ndarray) -> List[np.ndarray]:
    """Tüm ön işleme stratejilerini uygulayarak kenar/ikili haritalar listesi oluşturur."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return [
        _pre_soft(gray.copy()),
        _pre_medium(gray.copy()),
        _pre_hard(gray.copy()),
        _pre_lab_color(bgr.copy())
    ]


# ---------------------- 2. Geometry Helpers (İyileştirildi) ---------------------- #

def _order_points(pts: np.ndarray) -> np.ndarray:
    """Köşeleri Saat Yönünde Sıralar: tl, tr, br, bl."""
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Sol üst
    rect[2] = pts[np.argmax(s)]  # Sağ alt
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Sağ üst
    rect[3] = pts[np.argmax(diff)]  # Sol alt
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Verilen 4 noktaya göre görüntüye perspektif dönüşümü uygular."""
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect

    # Hedef en-boy oranını A4 kağıdına yakın tutalım (daha doğal görünüm)
    ratio = np.sqrt(2)  # A4 paper aspect ratio

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    # Görüntü yönünü (dikey/yatay) koru
    if maxHeight > maxWidth:
        dst_w, dst_h = int(800 / ratio), 800
    else:
        dst_w, dst_h = 1000, int(1000 / ratio)

    dst = np.array([
        [0, 0],
        [dst_w - 1, 0],
        [dst_w - 1, dst_h - 1],
        [0, dst_h - 1]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (dst_w, dst_h), flags=cv2.INTER_LANCZOS4)
    return warped


# ---------------------- 3. Candidate Generation (Sadeleştirildi ve Güçlendirildi) ---------------------- #

def find_document_candidates(binary_img: np.ndarray, ref_image: np.ndarray) -> List[np.ndarray]:
    """İkili görüntüdeki konturları analiz ederek dörtgen adayları bulur."""
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    H, W = ref_image.shape[:2]
    img_area = H * W
    candidates = []

    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        area = cv2.contourArea(c)
        if not (img_area * 0.1 < area < img_area * 0.95):
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            candidates.append(approx.astype(np.float32))

    return candidates


# ---------------------- 4. Scoring (Daha Akıllı Metrikler) ---------------------- #

def _get_angle(p1, p2, p3):
    """Üç noktadan oluşan açıyı hesaplar (p2 köşe)."""
    v1, v2 = p1 - p2, p3 - p2
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


def score_candidate(quad: np.ndarray, image: np.ndarray) -> float:
    """Bir dörtgen adayının 'doküman olma' olasılığını 0-1 arasında puanlar."""
    q = quad.reshape(4, 2)
    H, W = image.shape[:2]

    # 1. Alan Skoru: Çok küçük veya çok büyük olmamalı.
    area = cv2.contourArea(q)
    area_score = 1.0 if (H * W * 0.1) < area < (H * W * 0.95) else 0.0

    # 2. Açı Skoru: Köşeler 90 dereceye ne kadar yakın?
    angles = [_get_angle(q[i], q[(i + 1) % 4], q[(i + 2) % 4]) for i in range(4)]
    angle_score = np.mean([np.exp(-((a - 90) ** 2) / (2 * 15 ** 2)) for a in angles])

    # 3. Paralellik Skoru: Karşılıklı kenarlar ne kadar paralel?
    rect = _order_points(q)
    (tl, tr, br, bl) = rect

    def get_vec_angle(p1, p2): v = p2 - p1; return np.arctan2(v[1], v[0])

    angle_top = get_vec_angle(tl, tr)
    angle_bottom = get_vec_angle(bl, br)
    angle_left = get_vec_angle(tl, bl)
    angle_right = get_vec_angle(tr, br)

    parallel_score1 = np.exp(-((angle_top - angle_bottom) ** 2) / 0.1)
    parallel_score2 = np.exp(-((angle_left - angle_right) ** 2) / 0.1)
    parallelism_score = (parallel_score1 + parallel_score2) / 2.0

    # 4. En-Boy Oranı Skoru: Aşırı ince/uzun olmamalı.
    w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    h = (np.linalg.norm(tr - br) + np.linalg.norm(tl - bl)) / 2
    aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
    aspect_score = np.exp(-((aspect_ratio - 1.4) ** 2) / (2 * 0.5 ** 2))  # A4 ve benzeri oranları tercih et

    # Ağırlıklı Toplam Skor
    total_score = (
            0.15 * area_score +
            0.35 * angle_score +
            0.35 * parallelism_score +
            0.15 * aspect_score
    )
    return total_score


# ---------------------- 5. Post-processing (Ek İyileştirme) ---------------------- #

def finalize_image(image: np.ndarray) -> np.ndarray:
    """Düzeltilmiş görüntünün kontrastını ve okunabilirliğini artırır."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Kontrastı artır
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    # Hafif keskinleştirme
    sharpened = cv2.GaussianBlur(enhanced_gray, (0, 0), 3)
    sharpened = cv2.addWeighted(enhanced_gray, 1.5, sharpened, -0.5, 0)

    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


# ---------------------- Component (Ana Mantık) ---------------------- #

class PerspectiveCorrection(Component):
    """
    - Farklı ön işleme yöntemleriyle adaylar bulur, akıllı skorlama ile en iyisini seçer.
    - Düşük skorlu sonuçları reddederek güvenilirliği artırır.
    - Çıktı görüntüsünü okunabilirlik için son işlemden geçirir.
    """
    MIN_CONFIDENCE_SCORE = 0.60  # Güvenilir bir doküman tespiti için minimum skor

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
            raise ValueError("Input image empty or None")
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def run(self):
        img_obj = Image.get_frame(img=self.image, redis_db=self.redis_db)
        if img_obj is None or img_obj.value is None:
            raise ValueError("No input image provided or failed to load")
        src = self._prepare_image(img_obj.value)

        # 1. Adım: Farklı yöntemlerle birden çok ikili görüntü oluştur
        binary_maps = generate_edge_maps(src)

        # 2. Adım: Her haritadan aday dörtgenleri topla
        all_candidates = []
        for b_map in binary_maps:
            all_candidates.extend(find_document_candidates(b_map, src))

        if not all_candidates:
            # Hiç aday bulunamazsa, orijinal görüntüyü döndür (fallback)
            self.context["status"] = "No candidates found, returning original."
            warped = src
        else:
            # 3. Adım: Tüm adayları puanla ve en iyisini bul
            scored_candidates = [
                (score_candidate(c, src), c) for c in all_candidates
            ]
            best_score, best_quad = max(scored_candidates, key=lambda item: item[0])

            self.context["best_score"] = best_score

            # 4. Adım: En iyi adayın skoru yeterince yüksekse işlemi yap
            if best_score > self.MIN_CONFIDENCE_SCORE:
                self.context["status"] = "High-confidence document found and corrected."
                warped = four_point_transform(src, best_quad)
                warped = finalize_image(warped)  # Son iyileştirmeyi uygula
                self.context["src_quad"] = np.array(best_quad, dtype=float).tolist()
            else:
                # Güven skoru düşükse, orijinal görüntüyü kullanmak daha güvenli
                self.context[
                    "status"] = f"Best candidate score ({best_score:.2f}) is below threshold, returning original."
                warped = src
                self.context["src_quad"] = None

        # Sonuçları kaydet
        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)
        self.context["output_size"] = [int(warped.shape[1]), int(warped.shape[0])]
        self.context["method"] = {
            "pipelines": ["soft_adaptive", "medium_clahe_canny", "hard_unsharp_canny", "lab_color_grad"],
            "scorer_threshold": self.MIN_CONFIDENCE_SCORE,
        }
        return build_response(context=self)

Executor(sys.argv[1]).run()