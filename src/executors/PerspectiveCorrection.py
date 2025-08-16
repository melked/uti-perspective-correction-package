import os
import sys
import cv2
import numpy as np
from typing import List, Tuple, Optional

# --- SDK Entegrasyon Bölümü (Projenizdeki haliyle bırakılmıştır) --- #
# Bu kısımların projenizde doğru yapılandırıldığı varsayılmıştır.
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))
from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveCorrection.src.utils.response import build_response
from components.PerspectiveCorrection.src.models.PackageModel import PackageModel


# --------------------------------------------------------------------------------------
# BÖLÜM 1: GÖRÜNTÜ İŞLEME VE GEOMETRİ YARDIMCI FONKSİYONLARI
# Bu bölümdeki fonksiyonlar, temel ve tekrar eden görevleri yerine getirir.
# --------------------------------------------------------------------------------------

def _order_points(pts: np.ndarray) -> np.ndarray:
    """
    Dört noktadan oluşan bir diziyi standart bir sıraya koyar:
    [sol-üst, sağ-üst, sağ-alt, sol-alt].
    Bu sıralama, cv2.getPerspectiveTransform gibi fonksiyonların doğru çalışması için kritiktir.
    Sıralama, köşelerin (x, y) koordinat toplamları ve farklarına göre yapılır.

    Args:
        pts (np.ndarray): Şekli (4, 2) olan köşe noktaları dizisi.

    Returns:
        np.ndarray: Sıralanmış köşe noktaları.
    """
    # Fonksiyonunuz zaten bu işi en standart ve doğru şekilde yapıyordu, korunmuştur.
    pts = pts.reshape(4, 2).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Sol-üst: x+y toplamı en küçük olandır.
    rect[2] = pts[np.argmax(s)]  # Sağ-alt: x+y toplamı en büyük olandır.

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Sağ-üst: y-x farkı en küçük olandır.
    rect[3] = pts[np.argmax(diff)]  # Sol-alt: y-x farkı en büyük olandır.

    return rect


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Görüntünün 'pts' ile tanımlanan dörtgen bölgesini alıp, düz bir dikdörtgen haline getirir.
    Bu işlem "ileri yönlü perspektif dönüşümü" olarak bilinir.

    Args:
        image (np.ndarray): Orijinal görüntü.
        pts (np.ndarray): Düzeltilecek bölgenin 4 köşe noktası.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - warped (np.ndarray): Düzeltilmiş, dikdörtgen görüntü.
            - dst (np.ndarray): Düzeltilmiş görüntünün hedef köşe noktaları ([0,0], [width,0], ...).
                                Bu, ters dönüşüm için gereklidir.
    """
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect

    # Düzeltilmiş görüntünün genişliğini hesapla. Karşılıklı kenarların en uzun olanını al.
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    # Düzeltilmiş görüntünün yüksekliğini hesapla.
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    # Düzeltilmiş görüntünün hedef köşe noktalarını tanımla.
    # Bu, her zaman sol üstü (0,0) olan bir dikdörtgendir.
    dst = np.array([
        [0, 0], [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype=np.float32)

    # Perspektif dönüşüm matrisini hesapla ve uygula.
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LANCZOS4)

    return warped, dst


# --------------------------------------------------------------------------------------
# BÖLÜM 2: ÇOKLU STRATEJİLİ ÖN İŞLEME (ROBUSTNESS'IN TEMELİ)
# Tek bir yönteme güvenmek yerine, farklı senaryolar için tasarlanmış 4 farklı
# ön işleme tekniği kullanarak dokümanı bulma şansımızı maksimize ederiz.
# --------------------------------------------------------------------------------------

def _pre_soft(gray: np.ndarray) -> np.ndarray:
    """Strateji 1: Desenli veya karmaşık arka planlar için idealdir."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Adaptif eşikleme, aydınlatmanın homojen olmadığı durumlarda global eşiklemeden çok daha iyidir.
    # THRESH_BINARY_INV, genellikle aradığımız nesne (kağıt) arka plandan daha açık renkli olduğunda kullanılır.
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 21, 10)
    # Morfolojik kapama, metin veya desenler nedeniyle oluşan küçük delikleri ve kopuklukları birleştirir.
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    return th


def _pre_medium(gray: np.ndarray) -> np.ndarray:
    """Strateji 2: Genel amaçlı, kontrastı artırılmış kenar tespiti."""
    # CLAHE, görüntünün lokal kontrastını artırarak kenarların daha belirgin hale gelmesini sağlar.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    # Canny kenar tespiti, en popüler ve etkili kenar bulma algoritmalarından biridir.
    edges = cv2.Canny(eq, 60, 180, L2gradient=True)
    # Dilate, Canny tarafından bulunan kopuk kenarları birleştirmeye yardımcı olur.
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    return edges


def _pre_hard(gray: np.ndarray) -> np.ndarray:
    """Strateji 3: Düşük kontrastlı, uzaktan çekilmiş veya puslu görüntüler için tasarlanmıştır."""
    # Unsharp masking, görüntüyü bulanıklaştırıp orijinalden çıkararak kenarları keskinleştirir.
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    unsharp = cv2.addWeighted(gray, 1.8, blur, -0.8, 0)
    # Daha düşük Canny eşikleri, zayıf kenarların bile tespit edilmesini sağlar.
    edges = cv2.Canny(unsharp, 40, 120, L2gradient=True)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
    return edges


def _pre_color_suppressed(bgr: np.ndarray) -> np.ndarray:
    """Strateji 4: Renk bilgisini kullanarak en güvenilir sonucu üretme potansiyeline sahiptir."""
    # LAB renk uzayı, parlaklık (L) ve renk (A, B) bilgilerini ayırır.
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel, _, _ = cv2.split(lab)
    # Sadece parlaklık kanalına kontrast artırımı uygulayarak renklerin yanıltıcı etkisini ortadan kaldırırız.
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_channel)
    # Morfolojik gradyan, nesnelerin ana hatlarını (kenarlarını) vurgular.
    grad = cv2.morphologyEx(l_eq, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    # Otsu's thresholding, en uygun eşik değerini otomatik olarak bularak görüntüyü en iyi şekilde ikiye ayırır.
    _, th = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Son kapama işlemi, dokümanı tek ve solid bir nesne haline getirir.
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    return th


# --------------------------------------------------------------------------------------
# BÖLÜM 3: ADAY TESPİTİ, PUANLAMA VE SEÇİM (ALGORİTMANIN BEYNİ)
# Yüzlerce potansiyel şekil arasından en doğru olanı akıllıca seçen bölümdür.
# --------------------------------------------------------------------------------------

def _find_quad_candidates(binary_map: np.ndarray, min_area_ratio: float = 0.05) -> List[np.ndarray]:
    """
    Verilen bir ikili harita üzerinde 4 köşeli, dışbükey ve yeterince büyük konturları bulur.
    RETR_EXTERNAL, sadece en dıştaki konturları bularak iç içe geçmiş şekilleri görmezden gelir.
    """
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    min_area = binary_map.shape[0] * binary_map.shape[1] * min_area_ratio
    candidates = []

    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        if cv2.contourArea(c) < min_area:
            continue

        peri = cv2.arcLength(c, True)
        # approxPolyDP, bir konturun köşe sayısını azaltarak onu daha basit bir geometrik şekle benzetir.
        approx = cv2.approxPolyDP(c, 0.025 * peri, True)

        # Sadece 4 köşesi olan ve dışbükey (içe göçüksüz) olanları aday olarak kabul ediyoruz.
        if len(approx) == 4 and cv2.isContour_convex(approx):
            candidates.append(approx.astype(np.float32))

    return candidates


def _score_quad_candidate(quad: np.ndarray) -> float:
    """
    Bir dörtgen adayına 0 ile 1 arasında bir "doküman olma" puanı verir.
    Puanlama, geometrik ideallik üzerine kuruludur.
    """
    q = quad.reshape(4, 2)

    # 1. Açı Skoru (Ağırlık: %70): Bir dokümanın köşeleri 90 dereceye çok yakın olmalıdır.
    def get_angle(p1, p2, p3):
        v1, v2 = p1 - p2, p3 - p2
        cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    angles = [get_angle(q[i], q[(i + 1) % 4], q[(i + 2) % 4]) for i in range(4)]
    # Gaussian fonksiyonu: Açı 90'dan ne kadar uzaklaşırsa, skor o kadar yumuşak bir şekilde düşer.
    # Bu, "ya hep ya hiç" demek yerine esnek bir puanlama sağlar.
    angle_score = np.mean([np.exp(-((angle - 90) ** 2) / (2 * 10 ** 2)) for angle in angles])

    # 2. En-Boy Oranı Skoru (Ağırlık: %30): Dokümanlar genellikle aşırı ince veya uzun olmaz.
    rect = _order_points(q)
    w = (np.linalg.norm(rect[0] - rect[1]) + np.linalg.norm(rect[2] - rect[3])) / 2
    h = (np.linalg.norm(rect[1] - rect[2]) + np.linalg.norm(rect[0] - rect[3])) / 2
    aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
    # Bu skor, A4 kağıdına (oranı ~1.41) benzer oranları hafifçe ödüllendirir.
    aspect_score = np.exp(-((aspect_ratio - 1.4) ** 2) / (2 * 1.0 ** 2))

    return 0.7 * angle_score + 0.3 * aspect_score


# --------------------------------------------------------------------------------------
# BÖLÜM 4: RENK KORUMALI GÖRÜNTÜ İYİLEŞTİRME
# Perspektifi düzeltilmiş dokümanın okunabilirliğini, orijinal renklerini bozmadan artırır.
# --------------------------------------------------------------------------------------

def _enhance_color_preserving(image: np.ndarray) -> np.ndarray:
    """
    Düzeltilmiş görüntünün kontrastını, ORİJİNAL RENKLERİ KORUYARAK artırır.
    Bu, son isteğiniz olan renk koruma talebini karşılar.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l)
    merged_lab = cv2.merge([enhanced_l, a, b])
    enhanced_image = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    return enhanced_image


# --------------------------------------------------------------------------------------
# BÖLÜM 5: ANA KONTROL SINIFI (TÜM İSTEKLERİN BİRLEŞTİĞİ YER)
# Sizin orijinal Component yapınız korunarak, tüm mantık bu sınıf içinde yönetilir.
# --------------------------------------------------------------------------------------

class PerspectiveCorrection(Component):
    """
    Bu sınıf, tüm isteklerinizi karşılamak üzere tasarlanmış en detaylı ve sağlam
    perspektif düzeltme mantığını içerir.
    - Tüm adımlar detaylıca yorumlanmıştır.
    - Ayarlanabilir parametreler en üstte toplanmıştır.
    - Başarısızlık durumları (güven skoru düşüklüğü) zarifçe yönetilir.
    """

    # --- AYARLANABİLİR PARAMETRELER ---
    # Bu değerleri değiştirerek algoritmanın hassasiyetini ayarlayabilirsiniz.
    CONFIG = {
        "RESIZE_MAX_DIM": 1280,  # Performans için görüntünün indirgeneceği maksimum boyut.
        "MIN_AREA_RATIO": 0.05,  # Görüntünün %5'inden küçük alanlar doküman olarak kabul edilmez.
        "MIN_CONFIDENCE_SCORE": 0.70,  # Bir dörtgenin doküman olarak kabul edilmesi için gereken minimum puan.
    }

    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.context = {}
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        # Bu metodun projenizdeki amacına göre doldurulması gerekir.
        return {}

    def _prepare_and_resize(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Giriş görüntüsünü BGR'a çevirir, tipini doğrular ve gerekirse yeniden boyutlandırır."""
        if img is None or img.size == 0: return None
        if img.dtype != np.uint8: img = np.clip(img, 0, 255).astype(np.uint8)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        h, w = img.shape[:2]
        max_dim = self.CONFIG["RESIZE_MAX_DIM"]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        return img

    def run(self):
        """Ana operasyonun yürütüldüğü fonksiyon."""
        img_obj = Image.get_frame(img=self.image, redis_db=self.redis_db)
        if img_obj is None or img_obj.value is None:
            raise ValueError("Giriş görüntüsü yüklenemedi.")

        src_original = self._prepare_and_resize(img_obj.value)
        if src_original is None:
            raise ValueError("Giriş görüntüsü hazırlanamadı.")

        # --- Adım 1: Farklı stratejilerle potansiyel dokümanları bul ---
        gray = cv2.cvtColor(src_original, cv2.COLOR_BGR2GRAY)
        edge_maps = [
            _pre_soft(gray.copy()),
            _pre_medium(gray.copy()),
            _pre_hard(gray.copy()),
            _pre_color_suppressed(src_original.copy())
        ]

        # --- Adım 2: Tüm stratejilerden gelen adayları tek bir listede topla ---
        all_candidates = []
        for emap in edge_maps:
            all_candidates.extend(_find_quad_candidates(emap, self.CONFIG["MIN_AREA_RATIO"]))

        # --- Adım 3: Adayları puanla ve en iyisini seç ---
        best_quad, best_score = (None, 0.0)
        if all_candidates:
            scored_candidates = [(c, _score_quad_candidate(c)) for c in all_candidates]
            best_quad, best_score = max(scored_candidates, key=lambda item: item[1])

        self.context["best_score_found"] = float(best_score)

        # --- Adım 4: GÜVEN KONTROLÜ (En Kritik Adım!) ---
        # Eğer en yüksek skor bile belirlediğimiz eşiğin altındaysa, bu bir doküman değildir.
        # Hatalı bir düzeltme yapmaktansa, orijinal görüntüyü döndürmek en doğrusudur.
        # Bu, algoritmanın "her ortamda" güvenilir çalışmasını sağlar.
        if best_score < self.CONFIG["MIN_CONFIDENCE_SCORE"]:
            self.context[
                "status"] = f"Failure: No document found with confidence > {self.CONFIG['MIN_CONFIDENCE_SCORE']}"
            self.context["src_quad"] = None
            final_image = src_original
        else:
            # --- Adım 5: Başarılı Tespit Durumunda Düzeltme ve Yerleştirme ---
            self.context["status"] = "Success: Document corrected, enhanced, and overlaid with color preservation."
            self.context["src_quad"] = best_quad.tolist()

            # 5a. İleri yönlü warp ile dokümanı düzelt
            rect = _order_points(best_quad)
            warped_doc, dst_pts = _four_point_transform(src_original, rect)

            # 5b. Düzeltilmiş dokümanı RENKLERİ KORUYARAK iyileştir
            enhanced_doc = _enhance_color_preserving(warped_doc)

            # 5c. Geri yönlü warp ile iyileştirilmiş dokümanı orijinal yerine "eğ"
            inverse_matrix = cv2.getPerspectiveTransform(dst_pts, rect)
            h, w, _ = src_original.shape
            inversely_warped = cv2.warpPerspective(enhanced_doc, inverse_matrix, (w, h))

            # 5d. Maskeleme ile orijinal görüntü ve düzeltilmiş parçayı birleştir
            mask = np.zeros(src_original.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(mask, rect.astype(np.int32), 255)

            # Orijinal görüntüden doküman bölgesini "kesip" çıkar (siyah bir delik aç)
            background = cv2.bitwise_and(src_original, src_original, mask=cv2.bitwise_not(mask))
            # Düzeltilmiş dokümanın sadece ilgili bölgesini al
            foreground = cv2.bitwise_and(inversely_warped, inversely_warped, mask=mask)

            # Arka plan ve ön planı birleştirerek son görüntüyü oluştur
            final_image = cv2.add(background, foreground)

        # --- Adım 6: Sonuçları SDK formatında paketle ve döndür ---
        img_obj.value = final_image
        self.image = Image.set_frame(img_obj, package_uID=self.uID, redis_db=self.redis_db)
        self.context["output_size"] = [final_image.shape[1], final_image.shape[0]]
        self.context["config_used"] = self.CONFIG

        return build_response(context=self)

Executor(sys.argv[1]).run()