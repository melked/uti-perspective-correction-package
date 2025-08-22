import os
import sys
import cv2
import numpy as np
import math
from typing import Optional, List, Tuple, Dict

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveCorrection.src.utils.response import build_response
from components.PerspectiveCorrection.src.models.PackageModel import PackageModel


class PerspectiveCorrection(Component):
    class Params:
        def __init__(self, config=None):
            config = config or {}
            self.resize_height = config.get("resize_height", 800)
            self.min_area_ratio = config.get("min_area_ratio", 0.1)
            self.approx_poly_epsilon_ratio = config.get("approx_poly_epsilon_ratio", 0.02)
            self.min_confidence_threshold = config.get("min_confidence_threshold", 0.45)
            # Strategy Parameters
            self.hough_line_threshold = config.get("hough_line_threshold", 50)
            self.hough_min_line_length = config.get("hough_min_line_length", 50)
            self.hough_max_line_gap = config.get("hough_max_line_gap", 15)
            self.hough_angle_tolerance = config.get("hough_angle_tolerance", 10)
            # Preprocessing Parameters
            self.glare_threshold = config.get("glare_threshold", 235)
            self.shadow_clahe_clip_limit = config.get("shadow_clahe_clip_limit", 2.0)

    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.context = {}
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")
        params_data = self.request.get_param("params", {})
        self.params = self.Params(params_data)

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    # --- 1. GEOMETRY & UTILITY METHODS ---
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        pts = pts.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1);
        rect[0] = pts[np.argmin(s)];
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1);
        rect[1] = pts[np.argmin(diff)];
        rect[3] = pts[np.argmax(diff)]
        return rect

    def _four_point_transform(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        rect = self._order_points(pts)
        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl);
        widthB = np.linalg.norm(tr - tl)
        heightA = np.linalg.norm(tr - br);
        heightB = np.linalg.norm(tl - bl)
        maxWidth = int(max(widthA, widthB));
        maxHeight = int(max(heightA, heightB))
        dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LANCZOS4)

    # --- 2. ADVANCED PREPROCESSING & DIAGNOSTICS ---
    def _diagnose_image(self, image: np.ndarray) -> Dict:
        """Analyzes the image for key properties to guide preprocessing."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean, std_dev = np.mean(gray), np.std(gray)
        bright_pixels = np.sum(gray > self.params.glare_threshold) / gray.size
        diagnostics = {
            "contrast": "low" if std_dev < 45 else "normal",
            "has_glare": bright_pixels > 0.02,
        }
        print(f"Image Diagnostics: {diagnostics}")
        return diagnostics

    def _adaptive_preprocess(self, image: np.ndarray, diagnostics: Dict) -> np.ndarray:
        """Applies a tailored preprocessing pipeline based on image diagnostics."""
        processed = image.copy()

        if diagnostics["has_glare"]:
            print("   -> Glare detected, applying inpainting...")
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            _, glare_mask = cv2.threshold(gray, self.params.glare_threshold, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            glare_mask = cv2.dilate(glare_mask, kernel)
            processed = cv2.inpaint(processed, glare_mask, 3, cv2.INPAINT_TELEA)

        print("   -> Applying shadow & contrast compensation (LAB-CLAHE)...")
        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.params.shadow_clahe_clip_limit, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        processed = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        return processed

    # --- 3. EXPERT DETECTION STRATEGIES ---
    def _strategy_classic_contours(self, image_gray: np.ndarray) -> List[np.ndarray]:
        print("   -> Expert 1 (Classic Contours) running...")
        blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return self._get_quads_from_contours(contours)

    def _strategy_adaptive_thresh(self, image_gray: np.ndarray) -> List[np.ndarray]:
        print("   -> Expert 2 (Adaptive Threshold) running...")
        blurred = cv2.bilateralFilter(image_gray, 11, 17, 17)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return self._get_quads_from_contours(contours)

    def _get_quads_from_contours(self, contours: List[np.ndarray]) -> List[np.ndarray]:
        quads = []
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, self.params.approx_poly_epsilon_ratio * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                quads.append(approx.reshape(4, 2).astype(np.float32))
        return quads

    # --- 4. SCORING & SELECTION ---
    def _score_candidate(self, quad: np.ndarray, image_gray: np.ndarray) -> float:
        h, w = image_gray.shape
        total_area = h * w
        contour = quad.astype(np.int32)
        area = cv2.contourArea(contour)
        if not (self.params.min_area_ratio < area / total_area < 0.98): return 0.0

        M = cv2.moments(contour)
        if M["m00"] == 0: return 0.0
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        centrality = 1.0 - (np.linalg.norm(np.array([cx, cy]) - np.array([w / 2, h / 2])) / (max(w, h) / 2))

        width = (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2
        height = (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) / 2
        aspect_ratio = max(width, height) / (min(width, height) + 1e-6)
        aspect_score = 1.0 if 1.2 < aspect_ratio < 2.0 else 0.5

        mask = np.zeros(image_gray.shape, dtype="uint8")
        cv2.fillPoly(mask, [contour], 255)
        edged_content = cv2.Canny(image_gray, 50, 150)
        edge_pixel_count = np.count_nonzero(cv2.bitwise_and(edged_content, edged_content, mask=mask))
        content_score = min((edge_pixel_count / area) / 0.1, 1.0) if area > 0 else 0

        return (content_score * 0.5) + (centrality * 0.2) + (aspect_score * 0.2) + (area / total_area * 0.1)

    # --- 5. MAIN WORKFLOW ---
    def run(self):
        img_obj = Image.get_frame(img=self.image, redis_db=self.redis_db)
        if img_obj is None or img_obj.value is None: raise ValueError("Girdi görüntüsü alınamadı.")
        src_img_orig = img_obj.value
        if src_img_orig.dtype != np.uint8: src_img_orig = cv2.normalize(src_img_orig, None, 0, 255,
                                                                        cv2.NORM_MINMAX).astype(np.uint8)
        if src_img_orig.ndim == 2:
            src_img_orig = cv2.cvtColor(src_img_orig, cv2.COLOR_GRAY2BGR)
        elif src_img_orig.shape[-1] == 4:
            src_img_orig = cv2.cvtColor(src_img_orig, cv2.COLOR_BGRA2BGR)

        h_orig, w_orig = src_img_orig.shape[:2]
        ratio = h_orig / self.params.resize_height
        work_img = cv2.resize(src_img_orig, (int(w_orig / ratio), self.params.resize_height))

        # 1. Diagnose and Preprocess
        diagnostics = self._diagnose_image(work_img)
        processed_img = self._adaptive_preprocess(work_img, diagnostics)
        processed_img_gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)

        # 2. Generate Candidates with Expert Committee
        print("Hybrid Expert Committee is generating candidates...")
        candidates = []
        candidates.extend(self._strategy_classic_contours(processed_img_gray))
        candidates.extend(self._strategy_adaptive_thresh(processed_img_gray))

        # 3. Select the Best Candidate
        document_quad_orig = None
        warped = None
        if not candidates:
            print("No experts were able to find a candidate.")
        else:
            unique_candidates = []
            if candidates:
                unique_candidates.append(candidates[0])
                for cand in candidates[1:]:
                    if not any(np.allclose(cand, uc, atol=20) for uc in unique_candidates):
                        unique_candidates.append(cand)

            print(f"{len(unique_candidates)} unique candidates are being evaluated...")
            scored_candidates = [(self._score_candidate(q, processed_img_gray), q) for q in unique_candidates]
            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            best_score, best_quad_scaled = scored_candidates[0]

            if best_score > self.params.min_confidence_threshold:
                print(f"Best candidate selected with a score of {best_score:.2f}.")
                document_quad_orig = best_quad_scaled * ratio
                warped = self._four_point_transform(src_img_orig, document_quad_orig)
            else:
                print(
                    f"Best candidate's score ({best_score:.2f}) was below the minimum threshold ({self.params.min_confidence_threshold}).")

        # 4. Fallback and Finalize
        if warped is None:
            print("No valid document found. Using the original image.")
            warped = src_img_orig
            document_quad_orig = np.array([[0, 0], [w_orig - 1, 0], [w_orig - 1, h_orig - 1], [0, h_orig - 1]],
                                          dtype=np.float32)

        # 5. Save Output
        img_obj.value = warped
        self.image = Image.set_frame(img=img_obj, package_uID=self.uID, redis_db=self.redis_db)
        self.context["src_quad"] = document_quad_orig.tolist()
        self.context["output_size"] = [warped.shape[1], warped.shape[0]]
        return build_response(context=self)


if __name__ == "__main__":
    Executor(sys.argv[1]).run()