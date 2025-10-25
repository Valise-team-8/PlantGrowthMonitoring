import cv2
import numpy as np

def segment_plant(image):
    # Convert RGB to HSV (preprocessing.py converts BGR to RGB)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # Define green color range
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    result = cv2.bitwise_and(image, image, mask=mask)
    return mask, result
