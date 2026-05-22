import cv2
import numpy as np

img = cv2.imread('workers.png')

print(img.shape)

#########

# Scaling 
height, width = img.shape[:2]
scale = 0.5
resized_img = cv2.resize(img, (int(width * scale), int(height * scale)))
cv2.imwrite('resized_workers.png', resized_img)

# rotating
angle = 30
(h, w) = img.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated_img = cv2.warpAffine(img, M, (w, h))

cv2.Sobel

