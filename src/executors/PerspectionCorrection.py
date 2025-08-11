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


def reorder_corners(corners):
    new_corners = np.zeros((4, 2), dtype=corners.dtype)
    mean = np.mean(corners, axis=0)
    for corner in corners:
        if corner[0] < mean[0] and corner[1] < mean[1]:
            new_corners[0] = corner  # upper-left
        elif corner[0] > mean[0] and corner[1] < mean[1]:
            new_corners[1] = corner  # upper-right
        elif corner[0] > mean[0] and corner[1] > mean[1]:
            new_corners[2] = corner  # lower-right
        elif corner[0] < mean[0] and corner[1] > mean[1]:
            new_corners[3] = corner  # lower-left
    return new_corners


def preprocess_image(
    img,
    clahe_clip=3.0,
    clahe_grid=(8, 8),
    blur_kernel=(5, 5),
    gamma=1.0,
):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    enhanced = clahe.apply(gray)

    # Gamma correction
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
    gamma_corrected = cv2.LUT(enhanced, table)

    blurred = cv2.GaussianBlur(gamma_corrected, blur_kernel, 0)

    return blurred


def detect_corners(img, max_corners=10, quality=0.01, min_dist=30):
    corners = cv2.goodFeaturesToTrack(img, maxCorners=max_corners, qualityLevel=quality, minDistance=min_dist)
    if corners is not None:
        return np.squeeze(corners)
    else:
        return None


def correct_perspective(
    img,
    clahe_clip=3.0,
    clahe_grid=(8, 8),
    blur_kernel=(5, 5),
    gamma=1.0,
    canny_min=50,
    canny_max=150,
    max_corners=10,
    quality_level=0.01,
    min_distance=30,
    output_size_ratio=0.707,
    intermediate=False,
):
    processed = preprocess_image(img, clahe_clip, clahe_grid, blur_kernel, gamma)

    edges = cv2.Canny(processed, canny_min, canny_max)

    corners = detect_corners(edges, max_corners=max_corners, quality=quality_level, min_dist=min_distance)

    if corners is None or len(corners) < 4:
        raise ValueError("Yeterli köşe bulunamadı")

    # En uygun 4 köşeyi seç (örneğin en uzak 4 nokta)
    # Basitçe 4 köşe seçelim
    if len(corners) > 4:
        # Örnek basit seçim: görüntü merkezi ile en uzak 4 nokta
        center = np.mean(corners, axis=0)
        dists = np.linalg.norm(corners - center, axis=1)
        idxs = np.argsort(dists)[-4:]
        corners = corners[idxs]

    corners = reorder_corners(corners)

    h, w = img.shape[:2]
    min_dim = min(h, w)
    new_w, new_h = int(min_dim), int(min_dim * output_size_ratio)

    dst = np.float32([[0, 0], [new_w, 0], [new_w, new_h], [0, new_h]])

    M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    warped = cv2.warpPerspective(img, M, (new_w, new_h))

    if intermediate:
        from PIL import Image as PILImage
        return (
            PILImage.fromarray(processed),
            PILImage.fromarray(edges),
            PILImage.fromarray(warped),
        )
    else:
        from PIL import Image as PILImage
        return (PILImage.fromarray(warped),)


class PerspectiveCorrection(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.params = {
            "clahe_clip": self.request.get_param("clahe_clip") or 3.0,
            "clahe_grid": self.request.get_param("clahe_grid") or (8, 8),
            "blur_kernel": self.request.get_param("blur_kernel") or (5, 5),
            "gamma": self.request.get_param("gamma") or 1.0,
            "canny_min": self.request.get_param("canny_min") or 50,
            "canny_max": self.request.get_param("canny_max") or 150,
            "max_corners": self.request.get_param("max_corners") or 10,
            "quality_level": self.request.get_param("quality_level") or 0.01,
            "min_distance": self.request.get_param("min_distance") or 30,
            "output_size_ratio": self.request.get_param("output_size_ratio") or 0.707,
            "intermediate": self.request.get_param("intermediate") or False,
        }
        self.image = self.request.get_param("inputImage")

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

        corrected_tuple = correct_perspective(
            img_np,
            clahe_clip=self.params["clahe_clip"],
            clahe_grid=self.params["clahe_grid"],
            blur_kernel=self.params["blur_kernel"],
            gamma=self.params["gamma"],
            canny_min=self.params["canny_min"],
            canny_max=self.params["canny_max"],
            max_corners=self.params["max_corners"],
            quality_level=self.params["quality_level"],
            min_distance=self.params["min_distance"],
            output_size_ratio=self.params["output_size_ratio"],
            intermediate=self.params["intermediate"],
        )

        corrected_img = corrected_tuple[-1]

        img.value = np.array(corrected_img)

        self.image = Image.set_frame(img=img, package_uID=self.uID, redis_db=self.redis_db)

        packageModel = build_response(context=self)
        return packageModel


if __name__ == "__main__":
    Executor(sys.argv[1]).run()
