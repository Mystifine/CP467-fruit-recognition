import cv2
import numpy as np
from src.segment import segment


def colour_histogram(hsv, mask):
    """
    Calculate a 64-bin Hue-Saturation histogram
    using only the fruit pixels.
    """

    # 16 Hue bins and 4 Saturation bins
    hist = cv2.calcHist(
        [hsv],
        [0,1],
        mask,
        [16,4],
        [0,180, 0,256]
    )

    # Convert the histogram into 64 values
    hist = hist.flatten()

    # Normalize so all values add up to 1
    total = hist.sum()

    if total > 0:
        hist = hist / total

    return hist


def shape_features(contour):
    # Return zeros if no contour was found
    if contour is None:
        return np.zeros(3)

    # Find the rotated bounding rectangle
    rect = cv2.minAreaRect(contour)
    w, h = rect[1]

    if min(w, h) > 0:
        aspect_ratio = max(w,h) / min(w,h)
    else:
        aspect_ratio = 0

    # Calculate circularity
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if perimeter > 0:
        circularity = (4 * np.pi * area) / (perimeter ** 2)
    else:
        circularity = 0

    # Calculate solidity
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)

    if hull_area > 0:
        solidity = area / hull_area
    else:
        solidity = 0

    return np.array([aspect_ratio, circularity, solidity])



def hue_fractions(hsv, mask):
    # Get hue values from the HSV image
    hue = hsv[:, :, 0]

    # Only consider pixels belonging to the fruit
    fruit_pixels = hue[mask > 0]

    if len(fruit_pixels) == 0:
        return np.zeros(4)

    # Count pixels in each colour range
    red = np.sum((fruit_pixels < 10) | (fruit_pixels >= 160))
    orange = np.sum((fruit_pixels >= 10) & (fruit_pixels <= 21))
    yellow = np.sum((fruit_pixels >= 22) & (fruit_pixels <= 34))
    green = np.sum((fruit_pixels >= 35) & (fruit_pixels <= 84))

    # Calculate the fraction of each colour
    total = len(fruit_pixels)

    red = red / total
    orange = orange / total
    yellow = yellow / total
    green = green / total

    return np.array([red, orange, yellow, green])


def extract_features(img):
    # Separate the fruit from the background
    mask, contour = segment(img)

    # Convert the image from BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Extract color information
    hist = colour_histogram(hsv, mask)
    hue = hue_fractions(hsv, mask)

    # Extract shape information
    shape = shape_features(contour)

    return {
        "mask": mask,
        "contour": contour,
        "hist": hist,
        "shape": shape,
        "hue": hue
    }



if __name__ == "__main__":
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    cv2.circle(img, (50, 50), 30, (0, 165, 255), -1)
    features = extract_features(img)
    print("Histogram size:", features["hist"].shape)
    print("Histogram sum:", features["hist"].sum())
    print("Shape features:", features["shape"])
    print("Hue fractions:", features["hue"])
    print("Test completed!")
