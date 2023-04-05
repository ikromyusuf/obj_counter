import cv2
import numpy as np

# Load the image
img = cv2.imread('mouses.jpg')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Thresholding
_, thresh = cv2.threshold(gray, 187, 255, cv2.THRESH_BINARY)

# Morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40,40))
morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# Contour detection
contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Object counting
print(f"Sichqonchalar soni: {len(contours)}")

# Display the result

cv2.drawContours(img, contours, -1, (255, 0, 0), 2)
cv2.imshow('result', img)
# Display the result
cv2.waitKey(0)
cv2.destroyAllWindows()
