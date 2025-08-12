import os
import sys
import cv2
import numpy as np

# Assuming these imports are necessary for the Component structure,
# but the core perspective correction logic is self-contained.
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))
from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PerspectiveCorrection.src.utils.response import build_response
from components.PerspectiveCorrection.src.models.PackageModel import PackageModel


def order_points(pts):
    """Orders the points of a rectangle in top-left, top-right, bottom-right, bottom-left order."""
    # Initialize a list of 4 points
    rect = np.zeros((4, 2), dtype="float32")

    # The top-left point has the smallest sum, whereas the bottom-right point has the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    # Compute the difference between the points, the top-right point will have the smallest difference, while the bottom-left will have the largest
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect


def four_point_transform(image, pts):
    """Applies a four-point perspective transform to an image."""
    # Obtain a consistent order of the points and unpack them individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute the width of the new image, which will be the maximum distance between bottom-right and bottom-left x-coordinates or the top-right and top-left x-coordinates
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    # Compute the height of the new image, which will be the maximum distance between the top-right and bottom-right y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    # Now that we have the dimensions of the new image, construct the set of destination points to obtain a "birds eye view", (i.e. top-down view) of the image, again specifying points in the top-left, top-right, bottom-right, and bottom-left order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # Compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # Return the warped image
    return warped


def find_document_contour(image):
    """Finds the largest four-point contour in an image, likely representing a document."""
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur with a larger kernel to reduce noise more effectively, especially in blurry images
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Apply adaptive thresholding. Adjust block size and C constant for better results in varied lighting.
    # A larger block size can help in images with varying illumination.
    # The C constant is subtracted from the mean, affecting the threshold value.
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)

    # Apply morphological operations to close gaps and remove small noise artifacts.
    # Using slightly larger kernels here as well might help with blurry edges.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3) # Increased iterations
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=2)


    # Find contours in the processed image. RETR_LIST retrieves all contours, and CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments to their endpoints.
    contours, _ = cv2.findContours(opened.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Sort contours by area in descending order to prioritize larger contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    document_contour = None

    # Loop over the sorted contours
    for c in contours:
        # Approximate the contour with a polygon. 0.02 * peri is the epsilon value, a parameter that determines the maximum distance from the contour to the approximated polygon.
        # Adjusting epsilon might help with slightly irregular document shapes.
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True) # Slightly increased epsilon

        # If the approximated contour has four points (potentially a rectangle) and a reasonable area, consider it the document contour and break the loop
        if len(approx) == 4 and cv2.contourArea(approx) > 500: # Reduced minimum area slightly
            document_contour = approx
            break

    return document_contour


def correct_perspective(image):
    """Corrects the perspective of a document in an image."""
    # Find the document contour
    document_contour = find_document_contour(image)

    # If no suitable document contour is found, print a message and return the original image
    if document_contour is None:
        print("Could not find a suitable four-point contour.")
        return image

    # Apply the four point perspective transform using the found contour points
    warped = four_point_transform(image, document_contour.reshape(4, 2))

    return warped


class PerspectiveCorrection(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")
        self.params = self.request.get_param("params", None)

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def run(self):
        # Get the image from Redis and convert it to a NumPy array
        img = Image.get_frame(img=self.image, redis_db=self.redis_db)
        img_np = img.value
        # Ensure the image data type is uint8 (required by OpenCV)
        if img_np.dtype != np.uint8:
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)

        # Apply the perspective correction function
        result_img = correct_perspective(img_np) # Use the improved function

        # Update the image value in the Image object and store it back in Redis
        img.value = np.array(result_img)
        self.image = Image.set_frame(img=img, package_uID=self.uID, redis_db=self.redis_db)

        # Build and return the response
        return build_response(context=self)


if __name__ == "__main__":
    # Execute the component using the provided request string
    Executor(sys.argv[1]).run()