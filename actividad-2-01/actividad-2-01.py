import cv2
import pickle
import numpy as np

cap = cv2.VideoCapture('./actividad-2-01/video.mp4')

# Frame 10 - escala 1.5x
cap.set(cv2.CAP_PROP_POS_FRAMES, 9)
ret, frame = cap.read()
height, width = frame.shape[:2]
scale = 1.5
resized_img = cv2.resize(frame, (int(width * scale), int(height * scale)))
cv2.imwrite('frame_10.png', resized_img)

# Frame 30 - escala 0.5x
cap.set(cv2.CAP_PROP_POS_FRAMES, 29)
ret, frame = cap.read()
height, width = frame.shape[:2]
scale = 0.5
resized_img = cv2.resize(frame, (int(width * scale), int(height * scale)))
cv2.imwrite('frame_30.png', resized_img)

# Frame 50 - rotación 35 grados sobre su centro
cap.set(cv2.CAP_PROP_POS_FRAMES, 49)
ret, frame = cap.read()
(h, w) = frame.shape[:2]
center = (w // 2, h // 2)
cx, cy = center
angle = np.radians(35)
cos_a, sin_a = np.cos(angle), np.sin(angle)
M = np.array([
    [cos_a,  sin_a, (1 - cos_a) * cx - sin_a * cy],
    [-sin_a, cos_a,  sin_a * cx + (1 - cos_a) * cy]
], dtype=np.float64)
M_3x3 = np.vstack([M, [0, 0, 1]])
rotated_image = cv2.warpAffine(frame, M, (w, h))
cv2.imwrite('frame_50.png', rotated_image)

with open('rotation_matrix.pkl', 'wb') as f:
    pickle.dump(M_3x3, f)

# Frame 70 - flip horizontal
cap.set(cv2.CAP_PROP_POS_FRAMES, 69)
ret, frame = cap.read()
flipped_img = cv2.flip(frame, 1)
cv2.imwrite('frame_70.png', flipped_img)

cap.release()
print("Done.")