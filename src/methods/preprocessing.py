"""Load, resize, remove the white background, find the fruit contour."""
import cv2
import numpy as np

import config


def load_and_resize(path, size=None):
    size = size or config.IMG_SIZE
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def segment_fruit(img):
    """Binary mask (255 = fruit). The background is near-white, so we measure
    how far each pixel is from white using its darkest channel."""
    darkness = 255 - img.min(axis=2)                 # white -> 0, any colour -> larger
    darkness = cv2.GaussianBlur(darkness, (3, 3), 0)
    otsu_t, _ = cv2.threshold(darkness, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = min(max(otsu_t, 12), 40)                     # keep threshold in a sane range
    _, mask = cv2.threshold(darkness, t, 255, cv2.THRESH_BINARY)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    # keep only the largest blob and fill holes (e.g. bright highlights)
    contour = get_main_contour(mask)
    clean = np.zeros_like(mask)
    if contour is not None:
        cv2.drawContours(clean, [contour], -1, 255, thickness=cv2.FILLED)
    return clean


def get_main_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def process(path):
    """Convenience: path -> (img, mask, contour)."""
    img = load_and_resize(path)
    mask = segment_fruit(img)
    return img, mask, get_main_contour(mask)