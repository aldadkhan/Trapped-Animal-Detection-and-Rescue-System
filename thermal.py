import cv2
import numpy as np


def create_simulated_thermal(image):
    """
    Create a simulated thermal-style image from visible pixels.
    This does NOT measure temperature and cannot reveal hidden objects.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Expected a three-channel BGR visible-light image.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)


def process_thermal_frame(thermal_frame, percentile=90, minimum_area=20):
    """
    Find relatively bright regions in a thermal-camera image or intensity frame.
    This is image-intensity processing, not radiometric temperature analysis.
    """
    if thermal_frame is None or thermal_frame.size == 0:
        raise ValueError("Thermal frame is empty.")
    if not 0 < percentile < 100:
        raise ValueError("Percentile must be greater than 0 and less than 100.")
    if minimum_area < 0:
        raise ValueError("Minimum hotspot area cannot be negative.")

    if len(thermal_frame.shape) == 2:
        gray = thermal_frame.copy()
    elif len(thermal_frame.shape) == 3 and thermal_frame.shape[2] == 3:
        gray = cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Expected a grayscale or three-channel thermal image.")

    if gray.dtype != np.uint8:
        gray = cv2.normalize(
            gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )

    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    threshold_value = int(np.percentile(gray, percentile))
    if gray.min() == gray.max():
        mask = np.zeros_like(gray, dtype=np.uint8)
    elif threshold_value >= int(gray.max()):
        mask = np.where(gray >= threshold_value, 255, 0).astype(np.uint8)
    else:
        _, mask = cv2.threshold(
            gray, threshold_value, 255, cv2.THRESH_BINARY
        )

    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    if len(thermal_frame.shape) == 2:
        output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        output = thermal_frame.copy()

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    hotspots = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < minimum_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
        hotspots.append({
            "bbox": [x, y, x + w, y + h],
            "area": float(area)
        })

    hotspots.sort(key=lambda hotspot: hotspot["area"], reverse=True)
    return output, mask, hotspots
