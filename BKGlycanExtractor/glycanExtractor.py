import cv2
import numpy as np


def image_contour(img):
    # Convert to grayscale and apply Binary inverse thresholding
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, gray = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if contours:
        largest_index = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
    else:
        largest_index = None  # Handle cases where no contours are found

    return contours, largest_index


def crop_largest_component(img):
    contours, largest_index = image_contour(img)

    # Get the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(contours[largest_index])

    cropped_image = img[y:y+h, x:x+w]
    return cropped_image


def clean_largest_component(img):
    contours, largest_index = image_contour(img)

    out = np.zeros_like(img)
    cv2.drawContours(out, contours, largest_index, (255, 255, 255), -1)
    _, out = cv2.threshold(out, 230, 255, cv2.THRESH_BINARY_INV)

    cleaned_image = cv2.bitwise_or(out, img)
    return cleaned_image
